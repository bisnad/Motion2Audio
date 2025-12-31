from soundstream import from_pretrained as soundstream_from_pretrained
import torch
import torchaudio
import torch.nn.functional as nnF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np
import tqdm

soundstream_norm_path = "results_soundstream_dim32/soundstream_q_norm.pt"
soundstream_audio_sample_rate = 16000
min_soundstream_samples = 16000
hop_length = 960
max_windows_per_clip = 2000

"""
Create SoundStream Model
"""

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
    def wav_to_spec(self, wav_16k: torch.Tensor) -> torch.Tensor:
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
            
        #print("wav_16k s ", wav_16k.shape)

        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)
        wav_16k = wav_16k.to(self.device).float()
        
        #print("wav_16k 2 s ", wav_16k.shape)

        B, T_16k = wav_16k.shape
        if T_16k < self.min_len_16k:
            pad = self.min_len_16k - T_16k
            wav_16k = nnF.pad(wav_16k, (0, pad))
            
        #print("wav_16k 3 s ", wav_16k.shape)

        wav_16k = wav_16k.unsqueeze(1)  # [B,1,T_16k]
        
        #print("wav_16k 4 s ", wav_16k.shape)
        
        q = self._encode(wav_16k)       # [B,F,T_q]
        
        #print("q s ", q.shape)

        q_norm = (q - self._q_mean) / (self._q_std + 1e-8)
        
        #print("q_norm s ", q_norm.shape)
        
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

# calculate soundstream norm stats
class AudioWindowDataset(Dataset):
    def __init__(self, audio_tensors: list[torch.Tensor],
                 window_length: int, hop_length: int,
                 max_windows_per_clip: int | None = None):
        self.window_length = window_length
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
        win = w[s:s + self.window_length]
        # pad last window if slightly short
        if win.shape[0] < self.window_length:
            pad = self.window_length - win.shape[0]
            win = nnF.pad(win, (0, pad))
        return win

def compute_soundstream_norm_stats(
    backend: SoundStreamSpecBackend,
    audio_all_data: list[torch.Tensor],
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
    for batch_wav in tqdm.tqdm(loader, desc="[SS stats]"):
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
            wav_16k = nnF.pad(wav_16k, (0, pad))

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