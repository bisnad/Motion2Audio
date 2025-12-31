import os
import math
import numpy as np
import threading
import queue
import time
import sounddevice as sd
import torch
import torchaudio
from torch import nn
from torchaudio.functional import highpass_biquad

# osc specific imports
from pythonosc import dispatcher
from pythonosc import osc_server

# Music2Latent specific imports
from music2latent import EncoderDecoder

# mocap specific
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.pose_renderer import PoseRenderer

# -------------------------------------------------------
# Settings
# -------------------------------------------------------

# compute device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')

# mocap parameters
mocap_data_file = "E:/data/mocap/Diane/Solos/ZHdK_10.10.2025/fbx_50hz/trial-001.fbx"
mocap_pos_scale = 0.1
mocap_fps = 50
mocap_input_seq_length = 75

# audio parameters
print(sd.query_devices())
audio_data_file = "E:/data/audio/Diane/44khz/4d69949b.wav"
audio_output_device = 8
audio_sample_rate = 44100
audio_channels = 1
gen_buffer_size = 2048

audio_window_size = gen_buffer_size
audio_window_offset = audio_window_size // 2 # the correct value would be: audio_sample_rate // mocap_fps 
audio_samples_per_mocap_frame = audio_sample_rate // mocap_fps
audio_waveform_input_seq_length = int(audio_sample_rate / mocap_fps * mocap_input_seq_length)

audio_mel_filter_count = None # to be computed
audio_mel_input_seq_length = None # to be computed
audio_latents_input_seq_length = None # to be computed

play_buffer_size = audio_window_size * 16
playback_latency = 1.0
hpf_cutoff_hz = 15.0

# OSC parameters
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007

# FIFO Queue Settings
max_fifo_queue_length = 32 # 64

# Music2Latent Settings
latent_dim = 64 # fixed property of the Music2Latent model
audio_latents_input_seq_length = None # will be calculated
audio_dim = latent_dim

audio_latents_mean_file = "../motion_audio_transformer_music2latent/results_DianeFather/stat/latents_mean.pt"
audio_latents_std_file = "../motion_audio_transformer_music2latent/results_DianeFather/stat/latents_std.pt"

# Transformer parameters
transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 256
transformer_dropout = 0.1
pos_encoding_max_length = mocap_input_seq_length
transformer_weights_file = "../motion_audio_transformer_music2latent/results_DianeFather/weights/transformer_weights_epoch_400"


# -------------------------------------------------------
# Data Loading Functions
# -------------------------------------------------------

def load_mocap(file_path, scale):
    bvh_tools = bvh.BVH_Tools()
    fbx_tools = fbx.FBX_Tools()
    m_tools = mocap.Mocap_Tools()
    if file_path.lower().endswith(".bvh"):
        data = m_tools.bvh_to_mocap(bvh_tools.load(file_path))
        data["motion"]["rot_local"] = m_tools.euler_to_quat_bvh(data["motion"]["rot_local_euler"], data["rot_sequence"])
    else:
        data = m_tools.fbx_to_mocap(fbx_tools.load(file_path))[0]
        data["skeleton"]["offsets"] *= scale
        data["motion"]["pos_local"] *= scale
        data["skeleton"]["offsets"][0, [0, 2]] = 0.0
        data["motion"]["rot_local"] = m_tools.euler_to_quat(data["motion"]["rot_local_euler"], data["rot_sequence"])
    return data

def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    return waveform, sr

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    """Transformer mapping mocap sequence → audio latents."""
    def __init__(self, mocap_dim, audio_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, pos_encoding_max_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.mocap2embed = nn.Linear(mocap_dim, embed_dim)
        self.audio2embed = nn.Linear(audio_dim, embed_dim)
        self.positional_encoder = PositionalEncoding(dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_decoder_layers)
        self.embed2audio = nn.Linear(embed_dim, audio_dim)

    def get_src_mask(self, size):
        return torch.zeros(size, size)

    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, mocap_data, audio_data):
        src_mask = self.get_src_mask(mocap_data.shape[1]).to(mocap_data.device)
        tgt_mask = self.get_tgt_mask(audio_data.shape[1]).to(audio_data.device)
        mocap_embedded = self.positional_encoder(self.mocap2embed(mocap_data) * math.sqrt(self.embed_dim))
        audio_embedded = self.positional_encoder(self.audio2embed(audio_data) * math.sqrt(self.embed_dim))
        encoder_out = self.encoder(mocap_embedded, mask=src_mask)
        decoder_out = self.decoder(audio_embedded, encoder_out, tgt_mask=tgt_mask)
        return self.embed2audio(decoder_out)

# -------------------------------------------------------
# Audio Synthesis Helper Functions
# -------------------------------------------------------

@torch.no_grad()
def encode_audio(audio, music2latent_model):
    """Encode waveform to vocos latents."""
    
    #print("audio s ", audio.shape)
    
    audio_waveform = audio.to(torch.float16)
    audio_latents = music2latent_model.encode(audio_waveform)
    
    #print("audio_latents s ", audio_latents.shape)
    
    audio_latents = audio_latents.to(torch.float32)
    
    audio_latents = audio_latents.permute((0, 2, 1))
    
    #print("audio_latents 2 s ", audio_latents.shape)
    
    audio_latents = audio_latents.squeeze(0)
    
    #print("audio_latents 3 s ", audio_latents.shape)
    
    return audio_latents

@torch.no_grad()
def decode_audio(latent, music2latent_model):
    """Decode vocos latents to waveform."""
    
    #print("latent s ", latent.shape)
    
    audio_latents = latent.unsqueeze(0)
    audio_latents = audio_latents.permute((0, 2, 1))
    audio_latents = audio_latents.to(torch.float16)
    
    #print("audio_latents 2 s ", audio_latents.shape)
    
    audio_waveform = music2latent_model.decode(audio_latents)
    
    #print("audio_waveform s ", audio_waveform.shape)
    
    audio_waveform = audio_waveform.to(torch.float32)
    
    audio_waveform = audio_waveform.reshape(-1)
    
    #print("audio_waveform 2 s ", audio_waveform.shape)
    
    return audio_waveform

@torch.no_grad()
def audio_synthesis(mocap_seq, audio_latents):
    
    #print("audio_synthesis mocap_seq s ", mocap_seq.shape, " audio_latents s ", audio_latents.shape)
    
    """Predict next audio latents from mocap+current latents."""
    m_in = torch.tensor(mocap_seq, dtype=torch.float32).reshape(mocap_input_seq_length, -1).to(device)
    m_in = (m_in - mocap_mean) / (mocap_std + 1e-8)
    
    #print("audio_latents s ", audio_latents.shape)
    #print("audio_latents_mean s ", audio_latents_mean.shape)
    
    a_in = (audio_latents - audio_latents_mean) / (audio_latents_std + 1e-8)
    m_in = m_in.unsqueeze(0)
    a_in = a_in.unsqueeze(0)
    pred = transformer(m_in, a_in).squeeze(0)
    pred = pred * audio_latents_std + audio_latents_mean

    #print("audio_synth m_in s ", m_in.shape, " a_in s ", a_in.shape, " pred s ", pred.shape)

    return pred

# -------------------------------------------------------
# Load Audio and Mocap Data
# -------------------------------------------------------

mocap_data = load_mocap(mocap_data_file, mocap_pos_scale)
mocap_sequence = mocap_data["motion"]["rot_local"]
mocap_joint_count = mocap_sequence.shape[1]
mocap_joint_dim = mocap_sequence.shape[2]
mocap_pose_dim = mocap_joint_count * mocap_joint_dim
mocap_mean = torch.tensor(np.mean(mocap_sequence.reshape(mocap_sequence.shape[0], -1), axis=0, keepdims=True), dtype=torch.float32).to(device)
mocap_std = torch.tensor(np.std(mocap_sequence.reshape(mocap_sequence.shape[0], -1), axis=0, keepdims=True), dtype=torch.float32).to(device)
audio_waveform, _ = torchaudio.load(audio_data_file)
audio_waveform = audio_waveform[0]

# -------------------------------------------------------
# Load Models
# -------------------------------------------------------

encdec = EncoderDecoder()

audio_latents_input_sequence = encdec.encode(torch.rand(size=(1, audio_waveform_input_seq_length), dtype=torch.float32).to(device))
audio_latents_input_seq_length = audio_latents_input_sequence.shape[-1]

print("audio_latents_input_seq_length ", audio_latents_input_seq_length)

audio_latents_mean = torch.load(audio_latents_mean_file).to(device)
audio_latents_std = torch.load(audio_latents_std_file).to(device)

audio_latents_mean = audio_latents_mean.squeeze(2)
audio_latents_std = audio_latents_std.squeeze(2)

transformer = Transformer(
    mocap_sequence.shape[1]*mocap_sequence.shape[2],
    latent_dim,
    transformer_embed_dim,
    transformer_head_count,
    transformer_layer_count,
    transformer_layer_count,
    transformer_dropout,
    pos_encoding_max_length
).to(device)
transformer.load_state_dict(torch.load(transformer_weights_file, map_location=device))

# Precompute audio window
hann_window = torch.from_numpy(np.hanning(audio_window_size)).float().to(device)

# =========================================================
# FIFO Queues
# =========================================================

mocap_queue =queue.Queue(maxsize=max_fifo_queue_length)
audio_queue = queue.Queue(maxsize=max_fifo_queue_length)
export_audio_buffer = []

# =========================================================
# Motion Capture Receiver
# =========================================================

class MocapReceiver:
    """Receives mocap OSC messages and pushes frames to mocap_queue."""

    def __init__(self, ip, port):
        self.ip, self.port = ip, port
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/mocap/0/joint/rot_local", self.updateMocapQueue)
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
        self.thread = None

    def start(self):
        """Start OSC server in background thread."""
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def stop(self):
        """Gracefully shutdown OSC server."""
        self.server.shutdown()
        self.server.server_close()

    def join(self):
        """Wait for server thread to exit."""
        if self.thread is not None:
            self.thread.join()

    def updateMocapQueue(self, address, *args):
        """OSC handler for receiving mocap frames."""
        while mocap_queue.full():
            mocap_queue.get_nowait()  # Drop oldest
        mocap_frame = np.asarray(args, dtype=np.float32)
        mocap_frame = torch.tensor(mocap_frame, dtype=torch.float32).to(device)
        mocap_queue.put(mocap_frame)

# =========================================================
# Init Mocap and Audio Context
# =========================================================

input_waveform = audio_waveform[:audio_waveform_input_seq_length].unsqueeze(0).to(device)
input_audio_latents = encode_audio(input_waveform, encdec)

print("input_audio_latents s ", input_audio_latents.shape)

input_mocap_sequence = torch.zeros((mocap_input_seq_length, mocap_joint_count, mocap_joint_dim),
                                   dtype=torch.float32).to(device)
input_mocap_sequence[:, :, 0] = 1.0  # Example init
input_mocap_sequence = input_mocap_sequence.reshape((mocap_input_seq_length, mocap_pose_dim))

print("input_mocap_sequence s ", input_mocap_sequence.shape)

# debug
test_waveform = decode_audio(input_audio_latents, encdec)

test_waveform.shape

# =========================================================
# Audio Producer Thread
# =========================================================

shutdown_event = threading.Event()

mocap_idx = 0

def producer_thread():
    """Continuously predicts and enqueues new audio buffers."""
    global mocap_idx, input_audio_latents, input_mocap_sequence
    while not shutdown_event.is_set():
        if not audio_queue.full():
            input_mocap_sequence = torch.roll(input_mocap_sequence, shifts=-1, dims=0)
            if not mocap_queue.empty():
                input_mocap_sequence[-1] = mocap_queue.get_nowait().to(device)
            #print("input_mocap_sequence s ", input_mocap_sequence.shape, " input_latents s ", input_latents.shape)

            output_audio_latents = audio_synthesis(input_mocap_sequence, input_audio_latents)

            #print("output_audio_latents s ", output_audio_latents.shape)

            gen_waveform = decode_audio(output_audio_latents, encdec)
            
            #print("gen_waveform s ", gen_waveform.shape)
            
            gen_waveform = gen_waveform[-audio_window_size:]
            
            #print("gen_waveform 2 s ", gen_waveform.shape)
            
            # Optional highpass:
            # gen_waveform = highpass_biquad(gen_waveform, audio_sample_rate, hpf_cutoff_hz)
            audio_queue.put(gen_waveform.cpu().numpy())
            mocap_idx = (mocap_idx + 1) % mocap_sequence.shape[0]
            input_audio_latents = output_audio_latents.detach()
        else:
            time.sleep(0.01)  # Avoid busy-waiting

# =========================================================
# AUDIO CALLBACK (playback)
# =========================================================

# This must be maintained across audio_callback calls:
last_chunk = np.zeros(audio_window_size, dtype=np.float32)

def audio_callback(out_data, frames, time_info, status):
    """Overlap-add from audio_queue to output audio buffer."""
    global export_audio_buffer, last_chunk
    output = np.zeros((frames, audio_channels), dtype=np.float32)
    cursor = 0

    # Start with second half of last block from previous callback
    output[cursor:cursor+audio_window_offset, 0] += last_chunk[audio_window_offset:]

    while cursor < frames:
        try:
            chunk = audio_queue.get_nowait()
            chunk = chunk * hann_window.cpu().numpy()
        except queue.Empty:
            chunk = np.zeros(audio_window_size, dtype=np.float32)  # Output silence

        chunk_size = output[cursor:cursor+audio_window_size, 0].shape[0]
        output[cursor:cursor+chunk_size, 0] += chunk[:chunk_size]
        cursor += audio_window_offset
        last_chunk[:] = chunk[:]

    out_data[:] = output
    export_audio_buffer.append(output[:, 0])

# =========================================================
# MAIN RUNTIME
# =========================================================
if __name__ == "__main__":
    mocap_receiver = MocapReceiver(osc_receive_ip, osc_receive_port)
    mocap_receiver.start()
    threading.Thread(target=producer_thread, daemon=True).start()

    sd.sleep(2000)  # Allow queue to prefill

    with sd.OutputStream(
        samplerate=audio_sample_rate,
        device=audio_output_device,
        channels=audio_channels,
        callback=audio_callback,
        blocksize=play_buffer_size,
        latency=playback_latency
    ):
        print("Streaming audio... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopping...")
            shutdown_event.set()
            mocap_receiver.stop()
            mocap_receiver.join()

    # Save streamed audio to WAV
    export_audio_buffer_np = np.concatenate(export_audio_buffer, axis=0)
    export_audio_buffer_tensor = torch.from_numpy(export_audio_buffer_np).unsqueeze(0)
    torchaudio.save("audio_export.wav", export_audio_buffer_tensor, audio_sample_rate)