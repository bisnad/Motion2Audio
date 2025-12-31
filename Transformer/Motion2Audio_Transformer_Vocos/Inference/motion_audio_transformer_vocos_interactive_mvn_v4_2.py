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

# vocos specific
from vocos import Vocos

# mocap specific
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp
from common.pose_renderer import PoseRenderer

# -------------------------------------------------------
# Settings
# -------------------------------------------------------

# compute device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')

# mocap parameters
#mocap_data_file = "E:/data/mocap/Diane/Solos/ZHdK_10.10.2025/fbx_50hz/trial-001.fbx"
mocap_data_file = "E:/data/mocap/Eleni/Solos/ZHdK_04.12.2025/fbx_50hz/Eline_Session-001.fbx"
mocap_pos_scale = 0.1
mocap_fps = 50
mocap_input_seq_length = 68
#mocap_live_source = "MVN"
mocap_live_source = "MocapPlayer"

# audio parameters
print(sd.query_devices())
#audio_data_file = "E:/data/audio/Diane/48khz/4d69949b.wav"
audio_data_file = "E:/data/audio/Eleni/4_5870821179501060412.wav"
audio_output_device = 8
audio_sample_rate = 48000
audio_channels = 1
gen_buffer_size = 2048

audio_window_size = gen_buffer_size
audio_window_offset = audio_window_size // 2 # the correct value would be: audio_sample_rate // mocap_fps 
audio_waveform_input_seq_length = int(audio_sample_rate / mocap_fps * mocap_input_seq_length)
audio_samples_per_mocap_frame = audio_sample_rate // mocap_fps

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

# Vocos pre-trained model
vocos_pretrained_config = "kittn/vocos-mel-48khz-alpha1"

# Transformer parameters
transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 256
transformer_dropout = 0.1
pos_encoding_max_length = mocap_input_seq_length
transformer_weights_file = "../motion_audio_transformer_vocos/results_Eleni_Take1_v4_2/weights/transformer_weights_epoch_200"

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

# -------------------------------------------------------
# Model Definitions
# -------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # for batch-first: [1, max_len, dim_model]
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        
        #print("token_embedding s ", token_embedding.shape)
        #print("pos_encoding s ", self.pos_encoding.shape)
        
        # token_embedding: [batch_size, seq_len, dim_model]
        seq_len = token_embedding.size(1)
        # broadcast over batch dimension
        pe = self.pos_encoding[:, :seq_len, :]
        
        return self.dropout(token_embedding + pe)

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
# Mocap Format Conversion Helper Functions
# -------------------------------------------------------

# joint remapping from MVN to FBX

mvn2fbx_joint_index_map = [0, 15, 16, 17, 18, 19, 20, 21, 22, 1, 2, 3, 4, 11, 12, 13, 14, 7, 8, 9, 10, 5, 6]

def mvn2fbx_joint_index_remap(rotations_mvn):
    """
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations_mvn: (N, L, J, 4) tensor of unit quaternions describing the world rotations of each joint.
    Returns
     -- rotations_fbx: (N, L, J, 4) tensor of unit quaternions describing the world rotations of each joint.
    """
    
    # remap joint indices
    rotations_fbx =  rotations_mvn[:, :, mvn2fbx_joint_index_map, :]
    
    # Todo: possibly remap rotation dimensions
    
    return rotations_fbx
    

# joint remapping from FBX to MVN

fbx2mvn_joint_index_map = [0, 9, 10, 11, 12, 21, 22, 17, 18, 19, 20, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8]

def fbx2mvn_joint_index_remap(rotations_fbx):
    """
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations_fbx: (N, L, J, 4) tensor of unit quaternions describing the world rotations of each joint.
    Returns
     -- rotations_mvn: (N, L, J, 4) tensor of unit quaternions describing the world rotations of each joint.
    """
    
    # remap joint indices
    rotations_mvn = rotations_fbx[:, :, fbx2mvn_joint_index_map, :]
    
    # Todo: possibly remap rotation dimensions
    
    return rotations_mvn

# joint rotation conversion from local to world coordinates

def fbx2mvn_joint_rotation_conversion(rotations_local, offsets, parents):
    """
    Convert local joint rotations to world-space rotations using the kinematic tree.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations_local: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- offsets: (J, 3) tensor of joint offsets
     -- parents: list/array of length J, with -1 for root.
    Returns:
     -- rotations_world: (N, L, J, 4) tensor of unit quaternions describing world-space rotations.
    """

    assert len(rotations_local.shape) == 4
    assert rotations_local.shape[-1] == 4

    rotations_world = []

    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            # root: world rotation == local rotation
            rotations_world.append(rotations_local[:, :, 0])
        else:
            # child: world = parent_world * local
            rotations_world.append(
                qmul(rotations_world[parents[jI]], rotations_local[:, :, jI])
            )

    rotations_world = torch.stack(rotations_world, dim=2)  # (N, L, J, 4)
    return rotations_world

# joint rotation conversion from world to local coordinates

def mvn2fbx_joint_rotation_conversion(rotations_world, parents):
    """
    Convert world-space joint rotations (as a tensor) back to local-space rotations.

    Args:
        rotations: tensor of shape (N, L, J, 4) with world-space quaternions.
        parents: list/array of length J, with -1 for root.

    Returns:
        local_rotations_tensor: (N, L, J, 4) tensor of local rotations.
    """
    N, L, J, Q = rotations_world.shape
    assert Q == 4
    assert len(parents) == J

    def qconjugate(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    # Prepare output tensor
    rotations_local = torch.zeros_like(rotations_world)

    # Find root joint index
    root = None
    for j in range(J):
        if parents[j] == -1:
            root = j
            break
    if root is None:
        raise ValueError("No root joint found (parent == -1).")

    # 1) Root: local == world
    rotations_local[:, :, root, :] = rotations_world[:, :, root, :]

    # 2) Other joints
    for j in range(J):
        if j == root:
            continue
        p = parents[j]
        if p < 0:
            continue  # safety
        # parent inverse
        R_p_inv = qconjugate(rotations_world[:, :, p, :])
        R_j_world = rotations_world[:, :, j, :]

        # Flatten batch+time for qmul
        R_p_inv_flat = R_p_inv.reshape(-1, 4)
        R_j_world_flat = R_j_world.reshape(-1, 4)

        R_j_local_flat = qmul(R_p_inv_flat, R_j_world_flat)
        R_j_local = R_j_local_flat.view(N, L, 4)

        rotations_local[:, :, j, :] = R_j_local

    return rotations_local

# -------------------------------------------------------
# Audio Synthesis Helper Functions
# -------------------------------------------------------

@torch.no_grad()
def encode_audio(audio_waveform, vocos_model):
    
    #print("encode_audio audio_waveform s ", audio_waveform.shape)
    
    vocos_input = audio_waveform.reshape(1, -1).float().to(device)
    
    #print("vocos_input s ", vocos_input.shape)

    audio_mels = vocos_model.feature_extractor(vocos_input)
    
    #print("audio_mels s ", audio_mels.shape)

    return audio_mels[0]

@torch.no_grad()
def decode_audio(audio_mels, vocos_model):
    
    #print("decode_audio audio_mels s ", audio_mels.shape)
    
    audio_mels = audio_mels.unsqueeze(0)
    
    #print("decode_audio audio_mels 2 s ", audio_mels.shape)

    audio_waveform = vocos_model.decode(audio_mels)
    
    #print("audio_waveform s ", audio_waveform.shape)
    
    audio_waveform = audio_waveform.reshape(-1)
    
    #print("audio_waveform 2 s ", audio_waveform.shape)
    
    return audio_waveform


@torch.no_grad()
def audio_synthesis(mocap_seq, audio_mels):
    
    #print("audio_synthesis mocap_seq s ", mocap_seq.shape, " audio_mels s ", audio_mels.shape)
    
    """Predict next audio latents from mocap+current latents."""
    m_in = torch.tensor(mocap_seq, dtype=torch.float32).reshape(mocap_input_seq_length, -1).to(device)
    
    #print("m_in s ", m_in.shape)
    
    m_in = (m_in - mocap_mean) / (mocap_std + 1e-8)
    m_in = m_in.unsqueeze(0)
    
    #print("m_in 2 s ", m_in.shape)
    
    #print("audio_mels s ", audio_mels.shape)
    
    a_in = audio_mels.unsqueeze(0)
    
    #print("a_in s ", a_in.shape)
    
    a_in = (a_in - audio_mels_mean) / (audio_mels_std + 1e-8)
    a_in = a_in.permute((0, 2, 1))
    
    #print("a_in 2 s ", a_in.shape)
    
    a_out = transformer(m_in, a_in)
    
    #print("a_out s ", a_out.shape)
    
    a_out = a_out.permute((0, 2, 1))
    
    #print("a_out 2 s ", a_out.shape)
    
    pred_audio_mels = a_out * audio_mels_std + audio_mels_mean
    
    #print("pred_audio_mels s ", pred_audio_mels.shape)
    
    pred_audio_mels = pred_audio_mels.squeeze(0)
    
    #print("pred_audio_mels 2 s ", pred_audio_mels.shape)

    return pred_audio_mels

# -------------------------------------------------------
# Load Audio and Mocap Data
# -------------------------------------------------------

mocap_data = load_mocap(mocap_data_file, mocap_pos_scale)
mocap_sequence = mocap_data["motion"]["rot_local"]
mocap_offsets = torch.tensor(mocap_data["skeleton"]["offsets"], dtype=torch.float32).to(device)
mocap_parents = mocap_data["skeleton"]["parents"]
mocap_joint_count = mocap_sequence.shape[1]
mocap_joint_dim = mocap_sequence.shape[2]
mocap_pose_dim = mocap_joint_count * mocap_joint_dim

audio_waveform_data, _ = torchaudio.load(audio_data_file)
audio_waveform_data = audio_waveform_data[0]

# -------------------------------------------------------
# Load Vocos Model
# -------------------------------------------------------

vocos = Vocos.from_pretrained(vocos_pretrained_config).to(device).eval()
vocos.eval()

audio_mels_input_sequence = vocos.feature_extractor(torch.rand(size=(1, audio_waveform_input_seq_length), dtype=torch.float32).to(device))
audio_mels_input_seq_length = audio_mels_input_sequence.shape[-1]
audio_mel_filter_count = audio_mels_input_sequence.shape[1]

print("audio_mels_input_seq_length ", audio_mels_input_seq_length)
print("audio_mel_filter_count ", audio_mel_filter_count)

pos_encoding_max_length = max(mocap_input_seq_length, audio_mels_input_seq_length)
audio_dim = audio_mel_filter_count


# -------------------------------------------------------
# Compute Audio and Mocap Statistics
# -------------------------------------------------------

mocap_mean = torch.tensor(np.mean(mocap_sequence.reshape(mocap_sequence.shape[0], -1), axis=0, keepdims=True), dtype=torch.float32).to(device)
mocap_std = torch.tensor(np.std(mocap_sequence.reshape(mocap_sequence.shape[0], -1), axis=0, keepdims=True), dtype=torch.float32).to(device)

audio_mels = vocos.feature_extractor(audio_waveform_data.unsqueeze(0).to(device))
audio_mels_mean = torch.mean(audio_mels, dim=2, keepdim=True)
audio_mels_std = torch.std(audio_mels, dim=2, keepdim=True)

        
# -------------------------------------------------------
# Load Transformer Model
# -------------------------------------------------------

transformer = Transformer(
    mocap_sequence.shape[1]*mocap_sequence.shape[2],
    audio_dim,
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

class MocapReceiver_MocapPlayer:
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
        
        # set root rotation to zero
        mocap_frame = mocap_frame.reshape((mocap_joint_count, mocap_joint_dim))
        mocap_frame[0, :] = 0.0
        mocap_frame = mocap_frame.flatten()

        #print("mocap_frame ", mocap_frame)

        mocap_queue.put(mocap_frame)
        
class MocapReceiver_MVN:
    """Receives mocap OSC messages and pushes frames to mocap_queue."""

    def __init__(self, ip, port):
        self.ip, self.port = ip, port
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/mocap/0/joint/rot_world", self.updateMocapQueue)
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
        self.thread = None
        self.frame_counter = 0

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
    
    @torch.no_grad()
    def updateMocapQueue(self, address, *args):
        
        self.frame_counter += 1
        
        if self.frame_counter % 2 != 0:
            return
        
        """OSC handler for receiving mocap frames."""
        while mocap_queue.full():
            mocap_queue.get_nowait()  # Drop oldest
        mocap_frame_mvn = np.asarray(args, dtype=np.float32)
        mocap_frame_mvn = torch.tensor(mocap_frame_mvn)
        
        #print("mocap_frame_mvn s ", mocap_frame_mvn.shape)
        
        mocap_frame_mvn = mocap_frame_mvn.reshape((1, 1, mocap_joint_count, mocap_joint_dim)).to(device)
        
        #print("mocap_frame_mvn 2 s ", mocap_frame_mvn.shape)
        
        # joint index remap from mvn to fbx
        mocap_frame_fbx_world = mvn2fbx_joint_index_remap(mocap_frame_mvn)
        
        #print("mocap_frame_fbx_world s ", mocap_frame_mvn.shape)
        
        # joint rotation world to local conversion
        mocap_frame_fbx_local = mvn2fbx_joint_rotation_conversion(mocap_frame_fbx_world, mocap_parents)
        
        # swap quaternion coordinates
        mvn2fbx_quaternion_index_remap = [ 0, 2, 3, 1 ]
        mocap_frame_fbx_local = mocap_frame_fbx_local[:, :, :, mvn2fbx_quaternion_index_remap]
    
        """
        print("frame")
        for jI in range(mocap_joint_count):
            print("jI ", jI, " : ", mocap_frame_fbx_local[0, 0, jI, : ].cpu().numpy())
        """
        
        # set root rotation to zero
        mocap_frame_fbx_local = mocap_frame_fbx_local.reshape((mocap_joint_count, mocap_joint_dim))
        mocap_frame_fbx_local[0, :] = 0.0
    
        
        mocap_frame = mocap_frame_fbx_local.flatten()

        #print("mocap_frame ", mocap_frame)

        mocap_queue.put(mocap_frame)

# =========================================================
# Init Mocap and Audio Context
# =========================================================

input_waveform = audio_waveform_data[:audio_waveform_input_seq_length].to(device)
input_audio_mels = encode_audio(input_waveform, vocos)

print("input_audio_mels s ", input_audio_mels.shape)

"""
# debug
test_waveform = decode_audio(input_audio_mels, vocos)
"""


input_mocap_sequence = torch.zeros((mocap_input_seq_length, mocap_joint_count, mocap_joint_dim),
                                   dtype=torch.float32).to(device)
input_mocap_sequence[:, :, 0] = 1.0  # Example init
input_mocap_sequence = input_mocap_sequence.reshape((mocap_input_seq_length, mocap_pose_dim))

print("input_mocap_sequence s ", input_mocap_sequence.shape)

# =========================================================
# Audio Producer Thread
# =========================================================

shutdown_event = threading.Event()

mocap_idx = 0
save_counter = 0 # debug

def producer_thread():
    """Continuously predicts and enqueues new audio buffers."""
    global mocap_idx, input_audio_mels, input_mocap_sequence
    global save_counter # debug
    while not shutdown_event.is_set():
        if not audio_queue.full():
            input_mocap_sequence = torch.roll(input_mocap_sequence, shifts=-1, dims=0)
            if not mocap_queue.empty():
                input_mocap_sequence[-1] = mocap_queue.get_nowait().to(device)
                
            #print("input_mocap_sequence s ", input_mocap_sequence.shape)

            output_audio_mels = audio_synthesis(input_mocap_sequence, input_audio_mels)

            #print("output_audio_mels s ", output_audio_mels.shape)

            gen_waveform = decode_audio(output_audio_mels, vocos)
            
            """
            # debug begin
            if save_counter % 100 == 0 and save_counter < 1000:
                torchaudio.save("test_{}.wav".format(save_counter), gen_waveform.detach().cpu().unsqueeze(0), audio_sample_rate)
            save_counter += 1
            # debug end
            """
            
            #print("gen_waveform s ", gen_waveform.shape)
            
            gen_waveform = gen_waveform[-audio_window_size:]
            
            #print("gen_waveform 2 s ", gen_waveform.shape)
            
            # Optional highpass:
            # gen_waveform = highpass_biquad(gen_waveform, audio_sample_rate, hpf_cutoff_hz)
            audio_queue.put(gen_waveform.cpu().numpy())
            mocap_idx = (mocap_idx + 1) % mocap_sequence.shape[0]
            input_audio_mels = output_audio_mels.detach()
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
    output[cursor:cursor+audio_window_offset, 0] += last_chunk[-audio_window_offset:]

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
    
    if mocap_live_source == "MVN":
        mocap_receiver = MocapReceiver_MVN(osc_receive_ip, osc_receive_port)
    else:
        mocap_receiver = MocapReceiver_MocapPlayer(osc_receive_ip, osc_receive_port)
   
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