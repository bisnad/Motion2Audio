"""
Audio-only VAE on top of SoundStream.

This version is adapted so that:
- Training uses the same window hop behavior as inference
  (set audio_window_hop accordingly, e.g. half-overlap or 1-frame hop).
- Long VAE roundtrip uses overlap-add with the same hop,
  matching how you reconstruct long audio from overlapping windows.
"""

import os
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import torchaudio
import soundfile as sf
import auraloss

# -----------------------------
# CONFIG
# -----------------------------
audio_file_path = r"C:/Users/user/Documents/dance_generation/motion2audio/data"
audio_files = ["diane.wav"]
audio_sample_rate = 48000

results_dir = "results_soundstream_dim32"
os.makedirs(results_dir, exist_ok=True)

# optional pretrained VAE to resume
weight_file = r"vae_weights_ep5200.pt"

target_soundstream_sr = 16000
min_soundstream_samples = 16000

# VAE settings
latent_dim = 32
vae_conv_channel_counts = [16, 32, 64, 128]
vae_conv_kernel_size = (5, 3)
vae_dense_layer_sizes = [512]

vae_epochs = 10000
batch_size = 64
learning_rate = 1e-4

# Variational part: non-zero KL weight
beta_kl = 1e-3  # you can tune this; start small

# Windowing / hopping
audio_window_length = 48000          # 1 second @ 48 kHz

# <<< IMPORTANT: train hop == inference hop >>>
# - half overlap:
audio_window_hop = 960   # 24000
# - OR dense (1 mocap frame) you can try:
# audio_window_hop = audio_sample_rate // 50  # 960 for 50 fps

max_windows_per_clip = 2000

use_perc_loss = True
perc_loss_weight = 0.5

long_roundtrip_seconds = 30.0
long_roundtrip_window_length = audio_window_length
long_roundtrip_hop = audio_window_hop  # same hop as training

checkpoint_every = 50  # epochs

# -----------------------------
# Helpers
# -----------------------------
def save_wav_safe(path: str, wav: torch.Tensor, sr: int, normalize: bool = False):
    """
    Save waveform as float32, optional peak normalization.
    wav: [B,T] or [T]
    """
    x = wav.detach().cpu().float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        peak = x.abs().max()
        if peak > 0:
            x = x / peak * 0.95

    x = x.clamp(-1.0, 1.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x.t().numpy(), int(sr), subtype="FLOAT")
    print(f"[save_wav_safe] Saved {path} @ {sr} Hz")


def safe_audio_load(path: str, target_sr: int) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.size == 0:
        raise RuntimeError(f"Empty audio: {path}")
    w = torch.from_numpy(data.T)
    if w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    if sr != target_sr:
        w = torchaudio.functional.resample(w, sr, target_sr)
    print(f"[safe_audio_load] Loaded {os.path.basename(path)} with shape {tuple(w.shape)}")
    return w.contiguous()


# -----------------------------
# SoundStream Spec Backend
# -----------------------------
try:
    from soundstream import from_pretrained as soundstream_from_pretrained
except ImportError:
    soundstream_from_pretrained = None


class SoundStreamSpecBackend:
    """
    SoundStream wrapper:

    wav_to_spec: [B,T_48k] -> [B,F,T_q] (normalized codes)
    spec_to_wav: [B,F,T_q] (or [B,1,F,T_q]) -> [B,T_16k]
    """

    def __init__(
        self,
        device: torch.device,
        audio_sample_rate: int = 48000,
        target_sr: int = 16000,
        min_len_16k: int = 16000,
        q_mean: float | None = None,
        q_std: float | None = None,
    ):
        assert soundstream_from_pretrained is not None, \
            "pip install soundstream to use SoundStreamSpecBackend."

        self.device = device
        self.audio_sample_rate = audio_sample_rate
        self.orig_sample_rate = audio_sample_rate
        self.target_sr = target_sr
        self.min_len_16k = min_len_16k

        print("Loading SoundStream (NaturalSpeech2 config)...")
        self.model = soundstream_from_pretrained().to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.sample_rate = self.target_sr

        # Normalization stats (must come from dataset-level computation)
        if (q_mean is not None) and (q_std is not None):
            self._q_mean = torch.tensor(q_mean, device=self.device, dtype=torch.float32)
            self._q_std = torch.tensor(q_std, device=self.device, dtype=torch.float32).clamp_min(1e-3)
            print(f"[SoundStreamSpecBackend] Loaded q_mean={float(self._q_mean):.6f}, q_std={float(self._q_std):.6f}")
        else:
            self._q_mean = None
            self._q_std = None
            print("[SoundStreamSpecBackend] No normalization stats set yet.")
    def set_norm_stats(self, q_mean: float, q_std: float):
        """
        Set global normalization stats after computing them on the dataset.
        """
        self._q_mean = torch.tensor(q_mean, device=self.device, dtype=torch.float32)
        self._q_std = torch.tensor(q_std, device=self.device, dtype=torch.float32).clamp_min(1e-3)
        print(f"[SoundStreamSpecBackend] Normalization stats updated: "
              f"q_mean={float(self._q_mean):.6f}, q_std={float(self._q_std):.6f}")
    @torch.no_grad()
    def _encode(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """
        wav_16k: [B,1,T_16k]
        returns code-like tensor [B,F,T_q]
        """
        quantized = self.model(wav_16k, mode="encode")
        if isinstance(quantized, torch.Tensor):
            q = quantized
        elif isinstance(quantized, (list, tuple)):
            q_tensors = [
                qt if isinstance(qt, torch.Tensor) else torch.as_tensor(qt)
                for qt in quantized
            ]
            q = torch.cat(q_tensors, dim=1)
        else:
            raise RuntimeError(f"Unexpected quantized type: {type(quantized)}")

        if q.dim() == 2:
            q = q.unsqueeze(1)
        elif q.dim() == 3:
            pass
        else:
            raise RuntimeError(f"Unexpected quantized shape: {q.shape}")

        return q  # [B,F,T_q]

    @torch.no_grad()
    def wav_to_spec(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B,T] or [T] at audio_sample_rate
        returns: [B,F,T_q], normalized
        """
        if self._q_mean is None or self._q_std is None:
            raise RuntimeError(
                "SoundStreamSpecBackend normalization stats not set. "
                "Compute them once on the dataset and call set_norm_stats(), "
                "or pass q_mean/q_std into the constructor."
            )

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device).float()

        if self.audio_sample_rate != self.target_sr:
            wav_16k = torchaudio.functional.resample(
                wav, self.audio_sample_rate, self.target_sr
            )
        else:
            wav_16k = wav

        B, T_16k = wav_16k.shape
        if T_16k < self.min_len_16k:
            pad = self.min_len_16k - T_16k
            wav_16k = F.pad(wav_16k, (0, pad))

        wav_16k = wav_16k.unsqueeze(1)  # [B,1,T_16k]
        q = self._encode(wav_16k)       # [B,F,T_q]

        q_norm = (q - self._q_mean) / self._q_std
        return q_norm


    @torch.no_grad()
    def extract_spec(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.wav_to_spec(wav)     # [B,F,T]
        return spec.unsqueeze(1)

    @torch.no_grad()
    def spec_to_wav(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() == 4:
            spec = spec.squeeze(1)
        if self._q_mean is None or self._q_std is None:
            raise RuntimeError(
                "SoundStreamSpecBackend normalization stats not set. "
                "Cannot denormalize specs."
            )

        q = spec * self._q_std + self._q_mean
        q = q.to(self.device)

        wav_16k = self.model(q, mode="decode")  # [B,1,T]
        wav_16k = wav_16k.squeeze(1)
        return wav_16k


# -----------------------------
# Encoder / Decoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim, mel_count, mel_filter_count,
                 conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes

        stride = ((conv_kernel_size[0] - 1) // 2, (conv_kernel_size[1] - 1) // 2)
        padding = stride

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv2d(1, conv_channel_counts[0], conv_kernel_size,
                      stride=stride, padding=padding)
        )
        self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[0]))

        for i in range(1, len(conv_channel_counts)):
            self.conv_layers.append(
                nn.Conv2d(conv_channel_counts[i-1], conv_channel_counts[i],
                          conv_kernel_size, stride=stride, padding=padding)
            )
            self.conv_layers.append(nn.LeakyReLU(0.2))
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[i]))

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, mel_filter_count, mel_count)
            x = dummy
            for layer in self.conv_layers:
                x = layer(x)
            self.conv_out_shape = x.shape[1:]
            flat_dim = x.numel() // x.shape[0]

        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(nn.Linear(flat_dim, dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        for i in range(1, len(dense_layer_sizes)):
            self.dense_layers.append(
                nn.Linear(dense_layer_sizes[i-1], dense_layer_sizes[i])
            )
            self.dense_layers.append(nn.ReLU())

        self.fc_mu = nn.Linear(dense_layer_sizes[-1], latent_dim)
        self.fc_std = nn.Linear(dense_layer_sizes[-1], latent_dim)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.dense_layers:
            x = layer(x)
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        return mu, std

    def reparameterize(self, mu, std):
        return mu + std * torch.randn_like(std)


class Decoder(nn.Module):
    def __init__(self, latent_dim, mel_count, mel_filter_count,
                 conv_channel_counts, conv_kernel_size, dense_layer_sizes,
                 conv_out_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes

        stride = ((conv_kernel_size[0] - 1) // 2, (conv_kernel_size[1] - 1) // 2)
        padding = stride
        output_padding = (padding[0] - 1, padding[1] - 1)

        C0, H0, W0 = conv_out_shape
        flat_dim = C0 * H0 * W0
        self.base_shape = (C0, H0, W0)

        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(nn.Linear(latent_dim, dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        for i in range(1, len(dense_layer_sizes)):
            self.dense_layers.append(
                nn.Linear(dense_layer_sizes[i-1], dense_layer_sizes[i])
            )
            self.dense_layers.append(nn.ReLU())
        self.dense_layers.append(nn.Linear(dense_layer_sizes[-1], flat_dim))
        self.dense_layers.append(nn.ReLU())

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=self.base_shape)

        self.conv_layers = nn.ModuleList()
        for i in range(1, len(conv_channel_counts)):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[i-1]))
            self.conv_layers.append(
                nn.ConvTranspose2d(
                    conv_channel_counts[i-1], conv_channel_counts[i],
                    conv_kernel_size, stride=stride, padding=padding,
                    output_padding=output_padding
                )
            )
            self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        self.conv_layers.append(
            nn.ConvTranspose2d(
                conv_channel_counts[-1], 1,
                conv_kernel_size, stride=stride, padding=padding,
                output_padding=output_padding
            )
        )

    def forward(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        x = self.unflatten(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x[..., :self.mel_filter_count, :self.mel_count]
        return x  # [B,1,F,T]


# -----------------------------
# Dataset
# -----------------------------

def compute_soundstream_norm_stats(
    backend: SoundStreamSpecBackend,
    audio_all_data: List[torch.Tensor],
    window_length: int,
    hop_length: int,
    max_windows_per_clip: int | None = None,
    batch_size: int = 8,
    stats_path: str | None = None,
) -> tuple[float, float]:
    """
    Compute global mean/std of SoundStream codes q over the dataset (before normalization).

    Returns:
        (q_mean, q_std) as floats. Optionally saves them to stats_path.
    """
    # Build same window dataset as training
    dataset = AudioWindowDataset(
        audio_tensors=audio_all_data,
        window_length=window_length,
        hop_length=hop_length,
        max_windows_per_clip=max_windows_per_clip,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print("[compute_soundstream_norm_stats] Computing global q_mean/q_std...")
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    device = backend.device
    for batch_wav in tqdm(loader, desc="[SS stats]"):
        batch_wav = batch_wav.to(device).float()  # [B, 48000] at 48k

        # --- same preprocessing as wav_to_spec, but without normalization ---
        if backend.audio_sample_rate != backend.target_sr:
            wav_16k = torchaudio.functional.resample(
                batch_wav, backend.audio_sample_rate, backend.target_sr
            )
        else:
            wav_16k = batch_wav

        B, T_16k = wav_16k.shape
        if T_16k < backend.min_len_16k:
            pad = backend.min_len_16k - T_16k
            wav_16k = F.pad(wav_16k, (0, pad))

        wav_16k = wav_16k.unsqueeze(1)  # [B,1,T_16k]
        q = backend._encode(wav_16k)    # [B,F,T_q]

        q_flat = q.view(-1).float()
        total_sum += float(q_flat.sum().item())
        total_sq += float((q_flat * q_flat).sum().item())
        total_count += q_flat.numel()

    if total_count == 0:
        raise RuntimeError("No SoundStream codes found when computing stats.")

    q_mean = total_sum / total_count
    q_var = max(total_sq / total_count - q_mean * q_mean, 1e-6)
    q_std = float(np.sqrt(q_var))

    print(f"[compute_soundstream_norm_stats] q_mean={q_mean:.6f}, q_std={q_std:.6f}, count={total_count}")

    if stats_path is not None:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        torch.save({"q_mean": q_mean, "q_std": q_std}, stats_path)
        print(f"[compute_soundstream_norm_stats] Saved stats to {stats_path}")

    backend.set_norm_stats(q_mean, q_std)
    return q_mean, q_std

class AudioWindowDataset(Dataset):
    def __init__(self, audio_tensors: List[torch.Tensor],
                 window_length: int, hop_length: int,
                 max_windows_per_clip: int | None = None):
        self.windows = []
        for w in audio_tensors:
            w = w.squeeze(0)  # [T]
            T = w.shape[0]
            # include last valid start index as well
            starts = list(range(0, max(0, T - window_length + 1), hop_length))
            if max_windows_per_clip is not None:
                starts = starts[:max_windows_per_clip]
            for s in starts:
                self.windows.append((w, s))
        print(f"[AudioWindowDataset] Total windows: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w, s = self.windows[idx]
        win = w[s:s + audio_window_length]
        # pad last window if slightly short
        if win.shape[0] < audio_window_length:
            pad = audio_window_length - win.shape[0]
            win = F.pad(win, (0, pad))
        return win


# -----------------------------
# Loss / Training
# -----------------------------
perc_loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 8192],
    hop_sizes=[256, 512, 2048],
    win_lengths=[1024, 2048, 8192],
    scale="mel",
    n_bins=128,
    sample_rate=target_soundstream_sr,
    perceptual_weighting=True,
)


def vae_step(encoder, decoder, specs, backend, beta_kl=0.0, use_perc=False):
    """
    specs: [B,1,F,T]
    Proper VAE step: sampling + KL.
    """
    mu, std_raw = encoder(specs)
    std = F.softplus(std_raw) + 1e-8

    # Variational sampling in training
    z = encoder.reparameterize(mu, std)

    recon = decoder(z)

    rec = F.mse_loss(recon, specs)

    kl = -0.5 * torch.sum(
        1 + 2 * torch.log(std) - mu.pow(2) - std.pow(2),
        dim=-1
    ).mean()
    loss = rec + beta_kl * kl

    perc = torch.tensor(0.0, device=specs.device)
    if use_perc:
        with torch.no_grad():
            spec_gt = specs.squeeze(1)
            spec_rec = recon.squeeze(1)
        wav_gt = backend.spec_to_wav(spec_gt)
        wav_rec = backend.spec_to_wav(spec_rec)
        perc = perc_loss_fn(wav_rec.unsqueeze(1), wav_gt.unsqueeze(1))
        loss = loss + perc_loss_weight * perc

    return loss, rec.detach(), kl.detach(), perc.detach()


def train_vae_on_audio(
    epoch_start,
    encoder, decoder, backend,optim, audio_all_data,
    audio_window_length, audio_window_hop,
    device, epochs, batch_size,
    results_dir, long_roundtrip_seconds,
    long_roundtrip_window_length, long_roundtrip_hop,
    checkpoint_every=10,
):
    encoder.train()
    decoder.train()

    dataset = AudioWindowDataset(
        audio_tensors=audio_all_data,
        window_length=audio_window_length,
        hop_length=audio_window_hop,
        max_windows_per_clip=max_windows_per_clip,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)



    for ep in range(epoch_start, epochs):
        losses, recs, kls, percs = [], [], [], []
        pbar = tqdm(dataloader, desc=f"[VAE] epoch {ep+1}/{epochs}")
        for batch_wav in pbar:
            batch_wav = batch_wav.to(device)  # [B,48000]

            with torch.no_grad():
                specs = backend.extract_spec(batch_wav)

            optim.zero_grad(set_to_none=True)
            loss, rec, kl, perc = vae_step(
                encoder, decoder, specs, backend,
                beta_kl=beta_kl, use_perc=use_perc_loss
            )
            loss.backward()
            optim.step()

            losses.append(loss.item())
            recs.append(rec.item())
            kls.append(kl.item())
            percs.append(perc.item())

            pbar.set_postfix(
                loss=f"{np.mean(losses):.4f}",
                rec=f"{np.mean(recs):.4f}",
                kl=f"{np.mean(kls):.4f}",
                perc=f"{np.mean(percs):.4f}" if use_perc_loss else "n/a",
            )

        print(
            f"[VAE] epoch {ep+1}: "
            f"loss={np.mean(losses):.4f} rec={np.mean(recs):.4f} "
            f"kl={np.mean(kls):.4f}"
        )

        # Every N epochs: save checkpoint + long VAE roundtrip
        if (ep + 1) % checkpoint_every == 0:
            # checkpoint
            ckpt_path = os.path.join(results_dir, f"vae_weights_ep{ep+1}.pt")
            torch.save(
                {
                    "epoch": ep + 1,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "beta_kl": beta_kl,
                    "latent_dim": latent_dim,
                },
                ckpt_path,
            )
            print(f"[VAE] Saved checkpoint: {ckpt_path}")

            # long VAE roundtrip with same hop as training
            out_long = os.path.join(results_dir, f"vae_roundtrip_long_ep{ep+1}.wav")
            """
            vae_long_roundtrip(
                encoder, decoder, backend,
                waveform=audio_all_data[0],
                save_path=out_long,
                recon_seconds=long_roundtrip_seconds,
                window_length=long_roundtrip_window_length,
                window_hop=long_roundtrip_hop,
            )
            """

# -----------------------------
# Roundtrips
# -----------------------------
@torch.no_grad()
def soundstream_short_roundtrip(audio_all_data, backend, device, results_dir):
    test_audio = audio_all_data[0][:, :audio_window_length].to(device)
    spec = backend.wav_to_spec(test_audio)
    wav_rt = backend.spec_to_wav(spec)
    out_path = os.path.join(results_dir, "soundstream_roundtrip_short.wav")
    save_wav_safe(out_path, wav_rt, backend.sample_rate, normalize=False)

@torch.no_grad()
def soundstream_long_roundtrip(
    backend, waveform, save_path,
    recon_seconds=20.0,
    window_length=48000,
    window_hop=None,
):
    device = backend.device
    sr_orig = backend.audio_sample_rate  # 48000

    # Flatten to [T_48k]
    if isinstance(waveform, torch.Tensor):
        if waveform.dim() == 2:
            wav = waveform[0]
        else:
            wav = waveform
    else:
        wav = torch.as_tensor(waveform, dtype=torch.float32)
        if wav.dim() == 2:
            wav = wav[0]

    wav = wav.to(device)
    T = wav.shape[0]
    max_samples = min(T, int(recon_seconds * sr_orig))
    wav = wav[:max_samples]

    if window_hop is None:
        window_hop = window_length

    # Build 48k windows
    starts_48k = list(range(0, max(0, wav.shape[0] - window_length + 1), window_hop))
    if not starts_48k:
        print("[soundstream_long_roundtrip] No valid windows.")
        return

    windows_48k = []
    for s in starts_48k:
        win = wav[s:s + window_length]
        if win.shape[0] < window_length:
            pad = window_length - win.shape[0]
            win = F.pad(win, (0, pad))
        windows_48k.append(win)
    windows_48k = torch.stack(windows_48k, dim=0)  # [N, 48000]

    # Encode → decode in batch
    spec = backend.wav_to_spec(windows_48k)        # [N, F, T_q]
    wav_16k_batch = backend.spec_to_wav(spec)      # [N, T_16k]

    N, win_len_16k = wav_16k_batch.shape
    hop_16k = int(window_hop * backend.sample_rate / sr_orig)

    # Overlap-add in 16k
    env_16k = torch.hann_window(win_len_16k, device=device)
    total_len_16k = (N - 1) * hop_16k + win_len_16k
    rec_audio_16k = torch.zeros(total_len_16k, device=device)
    norm_16k = torch.zeros(total_len_16k, device=device)

    for i in range(N):
        start_16k = i * hop_16k
        end_16k = start_16k + win_len_16k
        rec_audio_16k[start_16k:end_16k] += wav_16k_batch[i] * env_16k
        norm_16k[start_16k:end_16k] += env_16k

    rec_audio_16k = rec_audio_16k / (norm_16k + 1e-8)

    save_wav_safe(save_path, rec_audio_16k.unsqueeze(0), backend.sample_rate, normalize=True)
    print(f"[soundstream_long_roundtrip] Saved long roundtrip to {save_path} "
          f"(len={rec_audio_16k.shape[0]} @ {backend.sample_rate} Hz)")



@torch.no_grad()
def vae_short_roundtrip(encoder, decoder, backend, audio_all_data, device, results_dir):
    """
    Debug: 1-second VAE roundtrip on the first training window.
    Uses mu (no sampling) for stable reconstruction.
    """
    encoder.eval()
    decoder.eval()
    test_audio = audio_all_data[0][:, :audio_window_length].to(device)  # [1,T]
    spec = backend.wav_to_spec(test_audio)         # [1,F,Tq]
    spec_vae = spec.unsqueeze(1)                   # [1,1,F,Tq]

    mu, std_raw = encoder(spec_vae)
    z = mu
    recon_spec = decoder(z).squeeze(1)             # [1,F,Tq]

    wav_rec = backend.spec_to_wav(recon_spec)      # [1,T16k]
    out_path = os.path.join(results_dir, "vae_roundtrip_short.wav")
    save_wav_safe(out_path, wav_rec, backend.sample_rate, normalize=True)


@torch.no_grad()
def vae_long_roundtrip(
    encoder, decoder, backend, waveform, save_path,
    recon_seconds: float = 10.0,
    window_length: int = 48000,   # 1s @ 48k
    window_hop: int | None = None # in 48k samples
):
    """
    Long VAE roundtrip over waveform, with optional overlap.

    - waveform: [1, T_48k] or [T_48k]
    - window_length: length of each 48k window (as in training)
    - window_hop: hop size in 48k samples. If None -> no overlap (hop=window_length)

    Steps:
        48k wav windows -> SoundStream spec -> VAE -> spec -> 16k wav
        -> overlap-add in 16k domain with Hann window.
    """
    device = backend.device

    was_training_enc = encoder.training
    was_training_dec = decoder.training
    encoder.eval()
    decoder.eval()

    # --- flatten waveform to 1D [T_48k] ---
    if isinstance(waveform, torch.Tensor):
        if waveform.dim() == 2:
            wav = waveform[0]  # [T]
        else:
            wav = waveform     # [T]
    else:
        wav = torch.as_tensor(waveform, dtype=torch.float32)
        if wav.dim() == 2:
            wav = wav[0]
    wav = wav.to(device)

    sr_orig = backend.audio_sample_rate  # 48000
    T = wav.shape[0]
    max_samples = min(T, int(recon_seconds * sr_orig))
    wav = wav[:max_samples]

    if window_hop is None:
        window_hop = window_length  # default: no overlap

    # ---- build window starts in 48k domain ----
    starts_48k = list(range(0, max(0, wav.shape[0] - window_length + 1), window_hop))
    if len(starts_48k) == 0:
        print("[vae_long_roundtrip] No valid windows, skipping.")
        if was_training_enc:
            encoder.train()
        if was_training_dec:
            decoder.train()
        return

    # ---- stack 48k windows into a batch [N, window_length] ----
    windows_48k = []
    for s in starts_48k:
        win = wav[s:s + window_length]
        if win.shape[0] < window_length:
            pad = window_length - win.shape[0]
            win = F.pad(win, (0, pad))
        windows_48k.append(win)
    windows_48k = torch.stack(windows_48k, dim=0)  # [N, 48000]

    # ---- SoundStream -> spec (batched) ----
    with torch.no_grad():
        spec = backend.wav_to_spec(windows_48k)  # [N,F,T_q]
    spec_vae = spec.unsqueeze(1)                # [N,1,F,T_q]

    # ---- VAE encode/decode (batched) ----
    with torch.no_grad():
        mu, std_raw = encoder(spec_vae)         # [N, latent_dim]
        std = F.softplus(std_raw) + 1e-8
        z = mu                                  # deterministic
        recon_spec = decoder(z).squeeze(1)      # [N,F,T_q]

    # ---- spec -> wav (batched, 16k) ----
    with torch.no_grad():
        wav_16k_batch = backend.spec_to_wav(recon_spec)  # [N,T_16k]

    N, win_len_16k = wav_16k_batch.shape
    hop_16k = int(window_hop * backend.sample_rate / sr_orig)

    # ---- overlap-add with Hann window ----
    env_16k = torch.hann_window(win_len_16k, device=device)
    total_len_16k = (N - 1) * hop_16k + win_len_16k

    rec_audio_16k = torch.zeros(total_len_16k, device=device)
    norm_16k = torch.zeros(total_len_16k, device=device)

    for i in range(N):
        start_16k = i * hop_16k
        end_16k = start_16k + win_len_16k
        rec_audio_16k[start_16k:end_16k] += wav_16k_batch[i] * env_16k
        norm_16k[start_16k:end_16k] += env_16k

    rec_audio_16k = rec_audio_16k / (norm_16k + 1e-8)

    save_wav_safe(save_path, rec_audio_16k.unsqueeze(0), backend.sample_rate, normalize=True)
    print(f"[vae_long_roundtrip] Saved VAE long roundtrip to {save_path} "
          f"(len={rec_audio_16k.shape[0]} @ {backend.sample_rate} Hz, "
          f"hop_48k={window_hop}, hop_16k={hop_16k})")

    # restore training flags
    if was_training_enc:
        encoder.train()
    if was_training_dec:
        decoder.train()
def soundstrean_with_io(device: torch.device,ss_norm_path: str | None = None,):
    q_mean = None
    q_std = None
    if ss_norm_path is not None and os.path.exists(ss_norm_path):
        stats = torch.load(ss_norm_path, map_location="cpu")
        q_mean = stats["q_mean"]
        q_std = stats["q_std"]
        print(f"[init_soundstream_vae_with_io] Loaded norm stats from {ss_norm_path}: "
              f"q_mean={q_mean:.6f}, q_std={q_std:.6f}")
    else:
        raise RuntimeError(
            f"[init_soundstream_vae_with_io] ss_norm_path not found: {ss_norm_path}. "
            "You must compute and save SoundStream normalization stats first."
        )

    backend = SoundStreamSpecBackend(
        device=device,
        audio_sample_rate=audio_sample_rate,      # uses your global config
        target_sr=target_soundstream_sr,
        min_len_16k=min_soundstream_samples,
        q_mean=q_mean ,
        q_std=q_std,
    )
    return backend
def init_soundstream_vae_with_io(
    device: torch.device,
    example_wav_48k: torch.Tensor,
    weight_file_path: str | None = None,
    ss_norm_path: str | None = None,   # <--- NEW: path to q_mean/q_std
):

    """
    Initialize SoundStream backend + VAE (Encoder/Decoder) and return
    convenient encode/decode functions.

    Args:
        device:
            torch.device("cuda") or torch.device("cpu").
        example_wav_48k:
            Example waveform at 48 kHz used to infer the SoundStream spec shape.
            Shape: [T] or [1, T] or [B, T] (only the first channel/batch is used).
        weight_file_path:
            Optional path to a VAE checkpoint (same format as in main()).
        ss_norm_path:
            Path to a .pt file containing {"q_mean": float, "q_std": float}
            computed once on the training set.

    Returns:
        backend, encoder, decoder, encode_wav_to_latent, decode_latent_to_wav, encode_decode_wav
    """
    # -------------------------
    # 0) Load normalization stats
    # -------------------------
    q_mean = None
    q_std = None
    if ss_norm_path is not None and os.path.exists(ss_norm_path):
        stats = torch.load(ss_norm_path, map_location="cpu")
        q_mean = stats["q_mean"]
        q_std = stats["q_std"]
        print(f"[init_soundstream_vae_with_io] Loaded norm stats from {ss_norm_path}: "
              f"q_mean={q_mean:.6f}, q_std={q_std:.6f}")
    else:
        raise RuntimeError(
            f"[init_soundstream_vae_with_io] ss_norm_path not found: {ss_norm_path}. "
            "You must compute and save SoundStream normalization stats first."
        )

    # -------------------------
    # 1) Backend (now with fixed q_mean/q_std)
    # -------------------------
    backend = SoundStreamSpecBackend(
        device=device,
        audio_sample_rate=audio_sample_rate,      # uses your global config
        target_sr=target_soundstream_sr,
        min_len_16k=min_soundstream_samples,
        q_mean=q_mean,
        q_std=q_std,
    )

    # Make sure example is shaped correctly and on device
    if example_wav_48k.dim() == 1:
        ex = example_wav_48k.unsqueeze(0)  # [1,T]
    else:
        ex = example_wav_48k
    ex = ex.to(device).float()

    # Use up to one training window for shape probing
    ex = ex[:, :audio_window_length]

    with torch.no_grad():
        spec_example = backend.wav_to_spec(ex)   # [1, F, T_q]

    mel_filter_count = spec_example.shape[1]     # F
    mel_count = spec_example.shape[2]            # T_q

    # -------------------------
    # 2) Build VAE
    # -------------------------
    encoder = Encoder(
        latent_dim=latent_dim,
        mel_count=mel_count,
        mel_filter_count=mel_filter_count,
        conv_channel_counts=vae_conv_channel_counts,
        conv_kernel_size=vae_conv_kernel_size,
        dense_layer_sizes=vae_dense_layer_sizes,
    ).to(device)

    vae_conv_channel_counts_rev = vae_conv_channel_counts.copy()[::-1]
    vae_dense_layer_sizes_rev = vae_dense_layer_sizes.copy()[::-1]

    decoder = Decoder(
        latent_dim=latent_dim,
        mel_count=mel_count,
        mel_filter_count=mel_filter_count,
        conv_channel_counts=vae_conv_channel_counts_rev,
        conv_kernel_size=vae_conv_kernel_size,
        dense_layer_sizes=vae_dense_layer_sizes_rev,
        conv_out_shape=encoder.conv_out_shape,
    ).to(device)

    # -------------------------
    # 3) Optional checkpoint load
    # -------------------------
    if weight_file_path is not None and os.path.exists(weight_file_path):
        ckpt = torch.load(weight_file_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
        print(f"[init_soundstream_vae_with_io] Loaded VAE weights from {weight_file_path}")
    else:
        print("[init_soundstream_vae_with_io] No checkpoint loaded; using random init.")

    encoder.eval()
    decoder.eval()

    # -------------------------
    # 4) High-level I/O helpers (unchanged)
    # -------------------------
    @torch.no_grad()
    def encode_wav_to_latent(wav_48k: torch.Tensor):
        if wav_48k.dim() == 1:
            x = wav_48k.unsqueeze(0)
        else:
            x = wav_48k
        x = x.to(device).float()

        spec = backend.extract_spec(x)   # [B,1,F,T_q]
        mu, std_raw = encoder(spec)
        std = F.softplus(std_raw) + 1e-8
        z = mu                           # deterministic code
        return z, mu, std, spec

    @torch.no_grad()
    def decode_latent_to_wav(z: torch.Tensor):
        z = z.to(device).float()
        recon_spec = decoder(z)                  # [B,1,F,T_q]
        wav_16k = backend.spec_to_wav(recon_spec)  # [B,T_16k]
        return wav_16k

    @torch.no_grad()
    def encode_decode_wav(wav_48k: torch.Tensor):
        z, mu, std, _spec_in = encode_wav_to_latent(wav_48k)
        wav_16k_rec = decode_latent_to_wav(z)
        return wav_16k_rec, z, mu, std

    return backend, encoder, decoder, encode_wav_to_latent, decode_latent_to_wav, encode_decode_wav

def test_soundstream(audio_file_path, audio_files, weight_file, results_dir, audio_window_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a test clip to probe shapes & test I/O
    test_path = os.path.join(audio_file_path, audio_files[0])
    test_audio = safe_audio_load(test_path, audio_sample_rate)  # [1,T]

    ss_norm_path = os.path.join(results_dir, "soundstream_q_norm.pt")

    backend, encoder, decoder, enc_fn, dec_fn, encdec_fn = init_soundstream_vae_with_io(
        device=device,
        example_wav_48k=test_audio,
        weight_file_path=weight_file,      # or None
        ss_norm_path=ss_norm_path,         # <--- NEW
    )

    # Full encode–decode:
    wav_16k_rec, z, mu, std = encdec_fn(test_audio[:, :audio_window_length])

    save_wav_safe(
        os.path.join(results_dir, "quick_encdec_test.wav"),
        wav_16k_rec,
        backend.sample_rate,
        normalize=True,
    )

"""
Audio weight

audio_weight_path=r"vae_weights_ep5200.pt"
test_soundstream(audio_file_path,audio_files,audio_weight_path,results_dir="results_soundstream_dim32",audio_window_length=48000)
"""
# -----------------------------
# MAIN
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 1) Load audio
    audio_all_data = []
    for fname in audio_files:
        full_path = os.path.join(audio_file_path, fname)
        audio = safe_audio_load(full_path, audio_sample_rate)
        audio_all_data.append(audio)

    # 2) Backend (without stats yet)
    backend = SoundStreamSpecBackend(
        device=device,
        audio_sample_rate=audio_sample_rate,
        target_sr=target_soundstream_sr,
        min_len_16k=min_soundstream_samples,
    )

    # 2b) Compute or load global normalization stats
    ss_stats_path = os.path.join(results_dir, "soundstream_q_norm.pt")
    if os.path.exists(ss_stats_path):
        stats = torch.load(ss_stats_path, map_location="cpu")
        backend.set_norm_stats(stats["q_mean"], stats["q_std"])
        print(f"[main] Loaded SoundStream norm stats from {ss_stats_path}")
    else:
        compute_soundstream_norm_stats(
            backend=backend,
            audio_all_data=audio_all_data,
            window_length=audio_window_length,
            hop_length=audio_window_hop,
            max_windows_per_clip=max_windows_per_clip,
            batch_size=8,
            stats_path=ss_stats_path,
        )

    # For realistic shape, use real audio
    with torch.no_grad():
        spec_example = backend.wav_to_spec(
            audio_all_data[0][:, :audio_window_length].to(device)
        )
    mel_filter_count = spec_example.shape[1]
    mel_count = spec_example.shape[2]
    print(f"Using SoundStream spec shape F x T = {mel_filter_count} x {mel_count}")

    # 3) Build VAE
    encoder = Encoder(
        latent_dim,
        mel_count,
        mel_filter_count,
        vae_conv_channel_counts,
        vae_conv_kernel_size,
        vae_dense_layer_sizes,
    ).to(device)

    vae_conv_channel_counts_rev = vae_conv_channel_counts.copy()[::-1]
    vae_dense_layer_sizes_rev = vae_dense_layer_sizes.copy()[::-1]

    decoder = Decoder(
        latent_dim,
        mel_count,
        mel_filter_count,
        vae_conv_channel_counts_rev,
        vae_conv_kernel_size,
        vae_dense_layer_sizes_rev,
        conv_out_shape=encoder.conv_out_shape,
    ).to(device)

    print(encoder)
    print(decoder)

    # 4) Baseline SoundStream-only roundtrips
    #soundstream_short_roundtrip(audio_all_data, backend, device, results_dir)

    ss_long_path = os.path.join(results_dir, "soundstream_roundtrip_long.wav")
    """
    soundstream_long_roundtrip(
        backend, waveform=audio_all_data[0],
        save_path=ss_long_path,
        recon_seconds=long_roundtrip_seconds,
        window_length=long_roundtrip_window_length,
        window_hop=long_roundtrip_hop,
    )
    """
    optim = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
    )
    # optional resume from previous VAE
    epoch_start = 0
    if weight_file and os.path.exists(weight_file):
        ckpt = torch.load(weight_file, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
        epoch_start = ckpt.get("epoch", 0)
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[main] Resumed VAE from {weight_file} at epoch {epoch_start}")

    # 5) Train VAE + periodic long roundtrips & checkpoints
    start = time.time()

    train_vae_on_audio(
        epoch_start,
        encoder, decoder, backend,optim,
        audio_all_data,
        audio_window_length, audio_window_hop,
        device, vae_epochs, batch_size,
        results_dir,
        long_roundtrip_seconds,
        long_roundtrip_window_length,
        long_roundtrip_hop,
        checkpoint_every=checkpoint_every,
    )

    print(f"[main] VAE training finished in {time.time() - start:.1f} s")

    # 6) Short VAE roundtrip (1 second, should be clearly audible)
    vae_short_roundtrip(encoder, decoder, backend, audio_all_data, device, results_dir)

    # 7) Final long VAE roundtrip (for the last epoch) with same hop as inference
    out_long = os.path.join(results_dir, f"vae_roundtrip_long_ep{vae_epochs}.wav")
    vae_long_roundtrip(
        encoder, decoder, backend,
        waveform=audio_all_data[0],
        save_path=out_long,
        recon_seconds=long_roundtrip_seconds,
        window_length=long_roundtrip_window_length,
        window_hop=long_roundtrip_hop,
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
