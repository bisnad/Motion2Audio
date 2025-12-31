"""
Imports
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import scipy.linalg as sclinalg
from sklearn.manifold import TSNE

import math
import random
import os, sys, time, subprocess
import numpy as np
import csv
import matplotlib.pyplot as plt
import json

# audio specific imports

import torchaudio
import torchaudio.transforms as transforms
import simpleaudio as sa
import auraloss

# vocos specific imports

from vocos import Vocos 

# motion specific imports

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Seed
"""

seed = 42 # if None, then no random seed will be used

"""
Mocap Settings
"""

mocap_data_path = "E:/Data/mocap/Eleni/Solos/ZHdK_04.12.2025/fbx_60hz"
mocap_data_files = ["Eline_Session-002.fbx"]
mocap_valid_ranges = [[780, 50163]]

mocap_pos_scale = 1.0
mocap_fps = 60
mocap_loss_weights_file = None
mocap_offset = 2

load_mocap_stats = False
mocap_mean_file = "results/stats/mocap_mean.pt"
mocap_std_file = "results/stats/mocap_std.pt"

"""
Audio Settings
"""

audio_data_path = "E:/data/audio/Eleni/"
audio_data_files = ["4_5870821179501060412.wav"]
audio_valid_ranges = [[2.71, 825.75]]

audio_sample_rate = 48000
audio_channels = 1
audio_waveform_offset = int(mocap_offset / mocap_fps * audio_sample_rate)
audio_waveform_length_per_mocap_frame = int(1.0 / mocap_fps * audio_sample_rate)

load_audio_stats = False
audio_mean_file = "results/stats/audio_mean.pt"
audio_std_file = "results/stats/audio_std.pt"

"""
Vocos Settings
"""

vocos_pretrained_config = "kittn/vocos-mel-48khz-alpha1"
audio_vocos_waveform_length = 44800

"""
Motion VAE Model Settings
"""

motion_vae_mocap_length = 8

motion_vae_latent_dim = 128
motion_vae_sequence_length = 24
motion_vae_rnn_layer_count = 2
motion_vae_rnn_layer_size = 512
motion_vae_dense_layer_sizes = [ 512 ]

"""
Audio VAE Model Settings
"""

audio_vae_waveform_length = int(motion_vae_mocap_length / mocap_fps * audio_sample_rate)
audio_vae_mel_count = None # automatically calculated

audio_vae_latent_dim = 128
audio_vae_conv_channel_counts = [ 16, 32, 64, 128 ]
audio_vae_conv_kernel_size = (5, 3)
audio_vae_dense_layer_sizes = [ 512 ]

"""
General Training Settings
"""

batch_size = 128
test_percentage  = 0.2
epochs = 400
model_save_interval = 50

vae_learning_rate = 1e-4 #1e-4
cl_learning_rate = 1e-5

vae_beta = 0.0 # will be calculated
vae_beta_cycle_duration = 100
vae_beta_min_const_duration = 20
vae_beta_max_const_duration = 20
vae_min_beta = 0.0
vae_max_beta = 0.2

cl_temperature = 0.07

"""
Motion VAE Training Settings
"""

motion_vae_norm_loss_scale = 0.1
motion_vae_pos_loss_scale = 0.1
motion_vae_quat_loss_scale = 1.0

load_motion_vae_weights = False
save_motion_vae_weights = True
motion_vae_encoder_weights_file = "results/weights/motion_vae_encoder_weights_epoch_600"
motion_vae_decoder_weights_file = "results/weights/motion_vae_decoder_weights_epoch_600"

"""
Audio VAE Training Settings
"""

audio_vae_rec_loss_scale = 5.0

load_audio_vae_weights = False
save_audio_vae_weights = True
audio_vae_encoder_weights_file = "results/weights/audio_vae_encoder_weights_epoch_600"
audio_vae_decoder_weights_file = "results/weights/audio_vae_decoder_weights_epoch_600"

"""
Mocap Visualisation Settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

"""
Fix Seeds
"""

def set_all_seeds(seed: int):
    # Python's built-in RNG
    random.seed(seed)
    # NumPy RNG
    np.random.seed(seed)
    # PyTorch RNG (CPU)
    torch.manual_seed(seed)
    # PyTorch RNG (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # (optional) PyTorch backend for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(seed)

"""
Load Data - Mocap
"""

# load mocap data
bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

mocap_all_data = []

for mocap_data_file, mocap_valid_range in zip(mocap_data_files, mocap_valid_ranges):

    if mocap_data_file.endswith(".bvh") or mocap_data_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_data_path + "/" + mocap_data_file)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif mocap_data_file.endswith(".fbx") or mocap_data_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_data_path + "/" + mocap_data_file)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only  
        
    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
    #print("pos_local shape", mocap_data["motion"]["pos_local"].shape)
    #print("rot_local_euler shape", mocap_data["motion"]["rot_local_euler"].shape)
    
    mocap_data["motion"]["pos_local"] = mocap_data["motion"]["pos_local"][mocap_valid_range[0]:mocap_valid_range[1], ...]
    mocap_data["motion"]["rot_local_euler"] = mocap_data["motion"]["rot_local_euler"][mocap_valid_range[0]:mocap_valid_range[1], ...]
    
    # set x and z offset of root joint to zero
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
    
    # set x and z offset of root joint to zero
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0
    
    if mocap_data_file.endswith(".bvh") or mocap_data_file.endswith(".BVH"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_data_file.endswith(".fbx") or mocap_data_file.endswith(".FBX"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

    mocap_all_data.append(mocap_data)
    
"""
Get Mocap Info
"""

mocap_skeleton = mocap_all_data[0]["skeleton"]

offsets = mocap_skeleton["offsets"].astype(np.float32)
parents = mocap_skeleton["parents"]
children = mocap_skeleton["children"]

mocap_motion = mocap_all_data[0]["motion"]["rot_local"]

mocap_joint_count = mocap_motion.shape[1]
mocap_joint_dim = mocap_motion.shape[2]
mocap_pose_dim = mocap_joint_count * mocap_joint_dim
mocap_dim = mocap_pose_dim

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

poseRenderer = PoseRenderer(edge_list)
    
"""
Calculate Mocap Stats
"""

if load_mocap_stats == True:
    mocap_mean = torch.load(mocap_mean_file)
    mocap_std = torch.load(mocap_std_file)
    
    mocap_mean = mocap_mean.to(device)
    mocap_std = mocap_std.to(device)
else:

    mocap_sequences_concat = [ mocap_data["motion"]["rot_local"] for mocap_data in mocap_all_data ]
    mocap_sequences_concat = np.concatenate(mocap_sequences_concat, axis=0)
    mocap_sequences_concat = mocap_sequences_concat.reshape(mocap_sequences_concat.shape[0], -1)
    
    mocap_mean = np.mean(mocap_sequences_concat, axis=0, keepdims=True)
    mocap_std = np.std(mocap_sequences_concat, axis=0, keepdims=True)
    
    mocap_mean = torch.from_numpy(mocap_mean).to(dtype=torch.float32)
    mocap_std = torch.from_numpy(mocap_std).to(dtype=torch.float32)
    
    print("mocap_mean s ", mocap_mean.shape)
    print("mocap_std s ", mocap_std.shape)
    
    torch.save(mocap_mean, mocap_mean_file)
    torch.save(mocap_std, mocap_std_file)
    
    mocap_mean = mocap_mean.to(device)
    mocap_std = mocap_std.to(device)

"""
Load Data - Audio
"""

audio_all_data = []

for audio_data_file, audio_valid_range in zip(audio_data_files, audio_valid_ranges):   
    
    audio_data, _ = torchaudio.load(audio_data_path + audio_data_file)
  
    print("audio_data s ", audio_data.shape)
    
    audio_range_begin = audio_valid_range[0]
    audio_range_end = audio_valid_range[1]
    
    print("audio_valid_range ", audio_valid_range)
    
    if audio_range_begin > 0 and audio_range_end > 0:
        audio_valid_range_sample = [ int(audio_valid_range[0] * audio_sample_rate), int(audio_valid_range[1] * audio_sample_rate)]    
    else: 
        audio_valid_range_sample = [ 0, audio_data.shape[-1]]   

    print("audio_valid_range_sample ", audio_valid_range_sample)
    
    audio_data = audio_data[:, audio_valid_range_sample[0]:audio_valid_range_sample[1]]
    
    print("audio_data 2 s ", audio_data.shape)
    
    audio_all_data.append(audio_data)

"""
Load Vocos Model
"""

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()

audio_vae_waveform = torch.zeros((1, audio_vae_waveform_length), dtype=torch.float32).to(device)
audio_vae_mels = vocos.feature_extractor(audio_vae_waveform)
audio_vae_waveform_rec = vocos.decode(audio_vae_mels)

print("audio_vae_waveform s ", audio_vae_waveform.shape)
print("audio_vae_mels s ", audio_vae_mels.shape)
print("audio_vae_waveform_rec s ", audio_vae_waveform_rec.shape)

audio_mel_filter_count = audio_vae_mels.shape[1]
audio_vae_mel_count = audio_vae_mels.shape[-1]

print("audio_mel_filter_count ", audio_mel_filter_count)
print("audio_vae_mel_count ", audio_vae_mel_count)
    
"""
for pose_count in range(8, 256):
    
    sample_count = pose_count * audio_waveform_length_per_mocap_frame
    
    dummy_wave = torch.zeros((1, sample_count)).to(device)
    dummy_mels = vocos.feature_extractor(dummy_wave)
    dummy_wave2 = vocos.decode(dummy_mels)

    if dummy_wave.shape[-1] == dummy_wave2.shape[-1]:
        print("pose_count ", pose_count)
        print("dummy_wave s ", dummy_wave.shape)
        print("dummy_mels s ", dummy_mels.shape)
"""

"""
Calculate Audio Stats
"""

if load_audio_stats == True:
    audio_mean = torch.load(audio_mean_file)
    audio_std = torch.load(audio_std_file)
    
    audio_mean = audio_mean.to(device)
    audio_std = audio_std.to(device)
else:
    
    audio_all_mels = []
    
    for audio_waveform in audio_all_data:
        
        #print("audio_waveform s ", audio_waveform.shape)
        
        audio_mels = vocos.feature_extractor(audio_waveform.to(device))
        
        #print("audio_mels s ", audio_mels.shape)
        
        audio_all_mels.append(audio_mels)
        
    audio_all_mels = torch.cat(audio_all_mels, axis=2)
    
    #print("audio_mels s ", audio_mels.shape)
    
    audio_mean = torch.mean(audio_all_mels, dim=2, keepdim=True)
    audio_std = torch.std(audio_all_mels, dim=2, keepdim=True)
    
    print("audio_mean s ", audio_mean.shape)
    print("audio_std s ", audio_std.shape)
    
    torch.save(audio_mean.detach().cpu(), audio_mean_file)
    torch.save(audio_std.detach().cpu(), audio_std_file)

"""
Create Dataset
"""

mocap_dataset = []
audio_dataset = []

audio_waveform_start_offset = audio_vocos_waveform_length - audio_vae_waveform_length
mocap_frame_start_offset = int(audio_waveform_start_offset / audio_sample_rate * mocap_fps)

for sI in range(len(mocap_all_data)):
    
    with torch.no_grad():
        
        mocap_data = mocap_all_data[sI]["motion"]["rot_local"].reshape(-1, mocap_pose_dim)
        audio_data = audio_all_data[sI][0]
    
        print(sI)
        print("mocap_data s ", mocap_data.shape)
        print("audio_data s ", audio_data.shape)
        
        mocap_frame_count = mocap_data.shape[0]

        for mfI in range(mocap_frame_start_offset, mocap_frame_count - motion_vae_mocap_length, mocap_offset):
            
            print("mfI ", mfI, " out of ", (mocap_frame_count - motion_vae_mocap_length))
            
            # mocap sequence part
            
            # get mocap sequence
            mocap_excerpt_start = mfI
            mocap_excerpt_end = mfI + motion_vae_mocap_length
            
            #print("mocap_excerpt_start ", mocap_excerpt_start)
            #print("mocap_excerpt_end ", mocap_excerpt_end)
            
            mocap_excerpt = mocap_data[mocap_excerpt_start:mocap_excerpt_end, :]
            mocap_excerpt = torch.from_numpy(mocap_excerpt).unsqueeze(0).to(torch.float32).to(device)
            
            #print("mfI ", mfI, " mocap_excerpt s ", mocap_excerpt.shape)
            
            # normalise mocap excerpt
            mocap_excerpt_norm = (mocap_excerpt - mocap_mean) / (mocap_std + 1e-8) 
            
            mocap_dataset.append(mocap_excerpt_norm.cpu())
            
            # audio sequence part
            asI = mfI * audio_waveform_length_per_mocap_frame
            audio_waveform_start = asI - audio_waveform_start_offset
            audio_waveform_end = audio_waveform_start + audio_vocos_waveform_length
            
            #print("audio_waveform_start ", audio_waveform_start)
            #print("audio_waveform_end ", audio_waveform_end)
            
            audio_waveform_excerpt = audio_data[audio_waveform_start:audio_waveform_end].unsqueeze(0).to(device)
            
            #print("mfI ", mfI, " audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
            
            audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
            
            #print("mfI ", mfI, " audio_mels_excerpt s ", audio_mels_excerpt.shape)
            
            # normalise audio mels excerpt
            audio_mels_excerpt_norm = (audio_mels_excerpt - audio_mean) / (audio_std + 1e-8)
            
            audio_dataset.append(audio_mels_excerpt_norm.cpu())

mocap_dataset = torch.cat(mocap_dataset, dim=0)
audio_dataset = torch.cat(audio_dataset, dim=0)

print("mocap_dataset s ", mocap_dataset.shape)
print("audio_dataset s ", audio_dataset.shape)


class MocapAudioDataset(Dataset):
    def __init__(self, mocap, audio):
        self.mocap = mocap
        self.audio = audio
    
    def __len__(self):
        return self.mocap.shape[0]
    
    def __getitem__(self, idx):
        return self.mocap[idx, ...], self.audio[idx, ...]
    
full_dataset = MocapAudioDataset(mocap_dataset, audio_dataset)

item_mocap, item_audio = full_dataset[0]

print("item_mocap s ", item_mocap.shape)
print("item_audio s ", item_audio.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

batch_mocap, batch_audio = next(iter(train_loader))

print("batch_mocap s ", batch_mocap.shape)
print("batch_audio s ", batch_audio.shape)

"""
Create Models
"""

"""
Create Motion VAE
"""

"""
Create Motion VAE Encoder
"""

class MotionEncoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(MotionEncoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.dense_layer_sizes = dense_layer_sizes
    
        # create recurrent layers
        rnn_layers = []
        rnn_layers.append(("encoder_rnn_0", nn.LSTM(self.pose_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        
        dense_layers = []
        
        dense_layers.append(("encoder_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))
            
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create final dense layers
            
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        x, (_, _) = self.rnn_layers(x)
        
        #print("x 2 ", x.shape)
        
        x = x[:, -1, :] # only last time step 
        
        #print("x 3 ", x.shape)
        
        x = self.dense_layers(x)
        
        #print("x 3 ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        #print("mu s ", mu.shape, " lvar s ", log_var.shape)
    
        return mu, std
    
    def reparameterize(self, mu, std):
            z = mu + std*torch.randn_like(std)
            return z
        
motion_encoder = MotionEncoder(motion_vae_mocap_length, mocap_pose_dim, motion_vae_latent_dim, motion_vae_rnn_layer_count, motion_vae_rnn_layer_size, motion_vae_dense_layer_sizes).to(device)

print(motion_encoder)

if load_motion_vae_weights and motion_vae_encoder_weights_file:
    motion_encoder.load_state_dict(torch.load(motion_vae_encoder_weights_file, map_location=device))
    
# test motion encoder

motion_encoder_input = batch_mocap.to(device)
motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(motion_encoder_input)
motion_encoder_output = motion_encoder.reparameterize(motion_encoder_output_mu, torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8)

print("motion_encoder_input s ", motion_encoder_input.shape)
print("motion_encoder_output s ", motion_encoder_output.shape)

"""
Create Motion VAE Decoder
"""

class MotionDecoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(MotionDecoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_count = rnn_layer_count
        self.dense_layer_sizes = dense_layer_sizes

        # create dense layers
        dense_layers = []
        
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
 
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create rnn layers
        rnn_layers = []

        rnn_layers.append(("decoder_rnn_0", nn.LSTM(self.dense_layer_sizes[-1], self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # final output dense layer
        final_layers = []
        
        final_layers.append(("decoder_dense_{}".format(dense_layer_count), nn.Linear(self.rnn_layer_size, self.pose_dim)))
        
        self.final_layers = nn.Sequential(OrderedDict(final_layers))
        
    def forward(self, x):
        #print("x 1 ", x.size())
        
        # dense layers
        x = self.dense_layers(x)
        #print("x 2 ", x.size())
        
        # repeat vector
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, self.sequence_length, 1)
        #print("x 3 ", x.size())
        
        # rnn layers
        x, (_, _) = self.rnn_layers(x)
        #print("x 4 ", x.size())
        
        # final time distributed dense layer
        x_reshaped = x.contiguous().view(-1, self.rnn_layer_size)  # (batch_size * sequence, input_size)
        #print("x 5 ", x_reshaped.size())
        
        yhat = self.final_layers(x_reshaped)
        #print("yhat 1 ", yhat.size())
        
        yhat = yhat.contiguous().view(-1, self.sequence_length, self.pose_dim)
        #print("yhat 2 ", yhat.size())

        return yhat

motion_vae_dense_layer_sizes_reversed = motion_vae_dense_layer_sizes.copy()
motion_vae_dense_layer_sizes_reversed.reverse()

motion_decoder = MotionDecoder(motion_vae_mocap_length, mocap_pose_dim, motion_vae_latent_dim, motion_vae_rnn_layer_count, motion_vae_rnn_layer_size, motion_vae_dense_layer_sizes_reversed).to(device)

print(motion_decoder)

if load_motion_vae_weights and motion_vae_decoder_weights_file:
    motion_decoder.load_state_dict(torch.load(motion_vae_decoder_weights_file, map_location=device))

# test motion decoder

motion_decoder_input = motion_encoder_output
motion_decoder_output = motion_decoder(motion_decoder_input)

print("motion_decoder_input s ", motion_decoder_input.shape)
print("motion_decoder_output s ", motion_decoder_output.shape)

"""
Create Audio VAE
"""

"""
Create Audio VAE Encoder
"""

class AudioEncoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)
        
        #print("conv_kernel_size ", conv_kernel_size)
        #print("stride ", stride)
        
        padding = stride
        
        self.conv_layers.append(nn.Conv2d(1, conv_channel_counts[0], self.conv_kernel_size, stride=stride, padding=padding))
        self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[0]))
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index]))

        self.flatten = nn.Flatten()
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        
        #print("last_conv_layer_size_x ", last_conv_layer_size_x)
        #print("last_conv_layer_size_y ", last_conv_layer_size_y)
        
        preflattened_size = [conv_channel_counts[-1], last_conv_layer_size_x, last_conv_layer_size_y]
        
        #print("preflattened_size ", preflattened_size)
        
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size_x * last_conv_layer_size_y
        
        #print("dense_layer_input_size ", dense_layer_input_size)
        #print("self.dense_layer_sizes[0] ", self.dense_layer_sizes[0])
        
        self.dense_layers.append(nn.Linear(dense_layer_input_size, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        # create final dense layers
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)


    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        for lI, layer in enumerate(self.conv_layers):
            
            #print("conv layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("conv layer ", lI, " x out ", x.shape)
    
        #print("x1 s ", x.shape)
        
        x = self.flatten(x)
        
        #print("x2 s ", x.shape)

        for lI, layer in enumerate(self.dense_layers):
            
            #print("dense layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("dense layer ", lI, " x out ", x.shape)
            
        #print("x3 s ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        #print("mu s ", mu.shape, " lvar s ", std.shape)

        return mu, std
    
    def reparameterize(self, mu, std):
        z = mu + std*torch.randn_like(std)
        return z

audio_encoder = AudioEncoder(audio_vae_latent_dim, audio_vae_mel_count, audio_mel_filter_count, audio_vae_conv_channel_counts, audio_vae_conv_kernel_size, audio_vae_dense_layer_sizes).to(device)

print(audio_encoder)

if load_audio_vae_weights and audio_vae_encoder_weights_file:
    audio_encoder.load_state_dict(torch.load(audio_vae_encoder_weights_file, map_location=device))

# test audio encoder

audio_encoder_input = batch_audio.unsqueeze(1).to(device)
audio_encoder_input = audio_encoder_input[:, :, :, -audio_vae_mel_count:]

audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
audio_encoder_output = audio_encoder.reparameterize(audio_encoder_output_mu, torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8)

print("audio_encoder_input s ", audio_encoder_input.shape)
print("audio_encoder_output s ", audio_encoder_output.shape)

"""
Create Audio VAE Decoder
"""

class AudioDecoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)
        
        print("stride ", stride)
                
        self.dense_layers.append(nn.Linear(latent_dim, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        
        #print("last_conv_layer_size_x ", last_conv_layer_size_x)
        #print("last_conv_layer_size_y ", last_conv_layer_size_y)
        
        preflattened_size = [conv_channel_counts[0], last_conv_layer_size_x, last_conv_layer_size_y]
        
        #print("preflattened_size ", preflattened_size)
        
        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size_x * last_conv_layer_size_y
        
        #print("dense_layer_output_size ", dense_layer_output_size)

        self.dense_layers.append(nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size))
        self.dense_layers.append(nn.ReLU())

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        padding = stride
        output_padding = (padding[0] - 1, padding[1] - 1) # does this universally work?
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index-1]))
            self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))

    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        for lI, layer in enumerate(self.dense_layers):
            
            #print("dense layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("dense layer ", lI, " x out ", x.shape)
            
        #print("x1 s ", x.shape)
        
        x = self.unflatten(x)
        
        #print("x2 s ", x.shape)

        for lI, layer in enumerate(self.conv_layers):
            
            #print("conv layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("conv layer ", lI, " x out ", x.shape)
    
        #print("x3 s ", x.shape)

        return x
    
audio_vae_conv_channel_counts_reversed = audio_vae_conv_channel_counts.copy()
audio_vae_conv_channel_counts_reversed.reverse()
    
audio_vae_dense_layer_sizes_reversed = audio_vae_dense_layer_sizes.copy()
audio_vae_dense_layer_sizes_reversed.reverse()

audio_decoder = AudioDecoder(audio_vae_latent_dim, audio_vae_mel_count, audio_mel_filter_count, audio_vae_conv_channel_counts_reversed, audio_vae_conv_kernel_size, audio_vae_dense_layer_sizes_reversed).to(device)

print(audio_decoder)

if load_audio_vae_weights and audio_vae_decoder_weights_file:
    audio_decoder.load_state_dict(torch.load(audio_vae_decoder_weights_file, map_location=device))

# test audio decoder
audio_decoder_input = audio_encoder_output
audio_decoder_output = audio_decoder(audio_decoder_input)

print("audio_decoder_input s ", audio_decoder_input.shape)
print("audio_decoder_output s ", audio_decoder_output.shape)

"""
Training
"""

def calc_vae_beta_values():
    
    vae_beta_values = []

    for e in range(epochs):
        
        cycle_step = e % vae_beta_cycle_duration
        
        #print("cycle_step ", cycle_step)

        if cycle_step < vae_beta_min_const_duration:
            vae_beta_value = vae_min_beta
            vae_beta_values.append(vae_beta_value)
        elif cycle_step > vae_beta_cycle_duration - vae_beta_max_const_duration:
            vae_beta_value = vae_max_beta
            vae_beta_values.append(vae_beta_value)
        else:
            lin_step = cycle_step - vae_beta_min_const_duration
            vae_beta_value = vae_min_beta + (vae_max_beta - vae_min_beta) * lin_step / (vae_beta_cycle_duration - vae_beta_min_const_duration - vae_beta_max_const_duration)
            vae_beta_values.append(vae_beta_value)
            
    return vae_beta_values

vae_beta_values = calc_vae_beta_values()

"""
Motion VAE Optimizer & Scheduler
"""

motion_vae_optimizer = torch.optim.Adam(list(motion_encoder.parameters()) + list(motion_decoder.parameters()), lr=vae_learning_rate)
motion_vae_scheduler = torch.optim.lr_scheduler.StepLR(motion_vae_optimizer, step_size=100, gamma=0.316) # reduce the learning every 100 epochs by a factor of 10

"""
Audio VAE Optimizer & Scheduler
"""

audio_vae_optimizer = torch.optim.Adam(list(audio_encoder.parameters()) + list(audio_decoder.parameters()), lr=vae_learning_rate)
audio_vae_scheduler = torch.optim.lr_scheduler.StepLR(audio_vae_optimizer, step_size=100, gamma=0.316) # reduce the learning every 100 epochs by a factor of 10

"""
Single Optimizer & Scheduler for both VAE Encoders
"""

contrastive_optimizer = torch.optim.Adam(
    list(motion_encoder.parameters()) + list(audio_encoder.parameters()), 
    lr=cl_learning_rate
)
contrastive_scheduler = torch.optim.lr_scheduler.StepLR(
    contrastive_optimizer, step_size=100, gamma=0.316
)

"""
General Loss Functions
"""

mse_loss = nn.MSELoss()
cross_entropy = nn.BCELoss()

# KL Divergence
def variational_loss(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #see also: see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #https://arxiv.org/abs/1312.6114
    vl=-0.5*torch.mean(1+ 2*torch.log(std)-mu.pow(2) -(std.pow(2)))
    return vl

# contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, emb_a, emb_b):
        # Normalize embeddings
        emb_a = nn.functional.normalize(emb_a, dim=-1)
        emb_b = nn.functional.normalize(emb_b, dim=-1)
        
        batch_size = emb_a.size(0)
        
        # Compute similarity matrix
        logits = torch.matmul(emb_a, emb_b.t()) / self.temperature
        
        # Labels for positive pairs (diagonal)
        labels = torch.arange(batch_size).to(emb_a.device)
        
        # Symmetric loss
        loss_a = nn.CrossEntropyLoss()(logits, labels)
        loss_b = nn.CrossEntropyLoss()(logits.t(), labels)
        
        return (loss_a + loss_b) / 2.0

contrastive_loss_fn = ContrastiveLoss(temperature=cl_temperature)

"""
Motion VAE Specific Loss Functions
"""

if mocap_loss_weights_file is not None:
    with open(mocap_loss_weights_file) as f:
        mocap_joint_loss_weights = json.load(f)
        mocap_joint_loss_weights = mocap_joint_loss_weights["joint_loss_weights"]
else:
    mocap_joint_loss_weights = [1.0]
    mocap_joint_loss_weights *= mocap_joint_count

mocap_joint_loss_weights = torch.tensor(mocap_joint_loss_weights, dtype=torch.float32)
mocap_joint_loss_weights = mocap_joint_loss_weights.reshape(1, 1, -1).to(device)

def motion_vae_norm_loss(yhat):
    
    _yhat = yhat.view(-1, 4)
    _norm = torch.norm(_yhat, dim=1)
    _diff = (_norm - 1.0) ** 2
    _loss = torch.mean(_diff)
    return _loss

def forward_kinematics(rotations, root_positions):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- root_positions: (N, L, 3) tensor describing the root joint positions.
    """

    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == 4
    
    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def motion_vae_pos_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim

    # normalize tensors
    _yhat = yhat.view(-1, 4)

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _y_rot = y.view((y.shape[0], y.shape[1], -1, 4))
    _yhat_rot = _yhat.view((y.shape[0], y.shape[1], -1, 4))

    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=True).to(device)

    _y_pos = forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = forward_kinematics(_yhat_rot, zero_trajectory)

    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    
    #print("_pos_diff s ", _pos_diff.shape)
    
    _pos_diff_weighted = _pos_diff * mocap_joint_loss_weights
    
    _loss = torch.mean(_pos_diff_weighted)

    return _loss

def motion_vae_quat_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    # normalize quaternion
    
    _y = y.view((-1, 4))
    _yhat = yhat.view((-1, 4))

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    
    # inverse of quaternion: https://www.mathworks.com/help/aeroblks/quaternioninverse.html
    _yhat_inv = _yhat_norm * torch.tensor([[1.0, -1.0, -1.0, -1.0]], dtype=torch.float32).to(device)

    # calculate difference quaternion
    _diff = qmul(_yhat_inv, _y)
    # length of complex part
    _len = torch.norm(_diff[:, 1:], dim=1)
    # atan2
    _atan = torch.atan2(_len, _diff[:, 0])
    # abs
    _abs = torch.abs(_atan)
    
    _abs = _abs.reshape(-1, motion_vae_mocap_length, mocap_joint_count)
    
    _abs_weighted = _abs * mocap_joint_loss_weights
    
    #print("_abs s ", _abs.shape)
    
    _loss = torch.mean(_abs_weighted)   
    return _loss

def motion_vae_loss(y_norm, yhat_norm, mu, std):
    
    # denormalise mocap data
    y = y_norm * mocap_std + mocap_mean
    yhat = yhat_norm * mocap_std + mocap_mean
    
    _norm_loss = motion_vae_norm_loss(yhat)
    _pos_loss = motion_vae_pos_loss(y, yhat)
    _quat_loss = motion_vae_quat_loss(y, yhat)

    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * motion_vae_norm_loss_scale
    _total_loss += _pos_loss * motion_vae_pos_loss_scale
    _total_loss += _quat_loss * motion_vae_quat_loss_scale
    _total_loss += _ae_kld_loss * vae_beta
    
    return _total_loss, _norm_loss, _pos_loss, _quat_loss, _ae_kld_loss

"""
Audio VAE Specific Loss Functions
"""

# Define perceptial loss: MR-STFT with perceptual mel weighting
perc_loss = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 8192],
    hop_sizes=[256, 512, 2048],
    win_lengths=[1024, 2048, 8192],
    scale="mel",          # use mel-scaled spectrograms
    n_bins=128,           # number of mel bins for the perceptual weighting
    sample_rate=48000,    # set to the actual SR used (48 kHz in your code)
    perceptual_weighting=True
)

def audio_vae_mel_loss(y_norm, yhat_norm):
    
    _aml = mse_loss(yhat_norm, y_norm)

    return _aml

def audio_vae_perc_loss(y_wave, yhat_wave):
    
    apl = perc_loss(yhat_wave, y_wave)
    
    return apl

"""
Training and Test Steps
"""    

"""
Motion VAE Training and Test Step
"""

def motion_vae_train_step(y_norm):
    
    #print("y_norm s ", y_norm.shape)
    
    motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(y_norm)
    mu = motion_encoder_output_mu
    std = torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8
    motion_decoder_input = motion_encoder.reparameterize(mu, std)
    
    #print("motion_decoder_input s ", motion_decoder_input.shape)
    
    yhat_norm = motion_decoder(motion_decoder_input)
    
    #print("yhat_norm s ", yhat_norm.shape)
    
    # denormalise mocap data
    y = y_norm * mocap_std + mocap_mean
    yhat = yhat_norm * mocap_std + mocap_mean
    
    #print("y s ", y.shape)
    #print("yhat s ", yhat.shape)
    
    # kld loss
    _kld_loss = variational_loss(mu, std)
    # quaternion normalisation loss
    _norm_loss = motion_vae_norm_loss(yhat)
    
    # position rec loss
    _pos_loss = motion_vae_pos_loss(y, yhat)
    
    # rotation rec loss
    _quat_loss = motion_vae_quat_loss(y, yhat)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * motion_vae_norm_loss_scale
    _total_loss += _pos_loss * motion_vae_pos_loss_scale
    _total_loss += _quat_loss * motion_vae_quat_loss_scale
    _total_loss += _kld_loss * vae_beta
    
    # Backpropagation
    motion_vae_optimizer.zero_grad()
    _total_loss.backward()
    
    motion_vae_optimizer.step()
    
    return _total_loss, _norm_loss, _pos_loss, _quat_loss, _kld_loss

"""
_losses = motion_vae_train_step(batch_mocap.to(device))
"""

@torch.no_grad()
def motion_vae_test_step(y_norm):

    motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(y_norm)
    mu = motion_encoder_output_mu
    std = torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8
    motion_decoder_input = motion_encoder.reparameterize(mu, std)
    yhat_norm = motion_decoder(motion_decoder_input)
    
    # denormalise mocap data
    y = y_norm * mocap_std + mocap_mean
    yhat = yhat_norm * mocap_std + mocap_mean
    
    # kld loss
    _kld_loss = variational_loss(mu, std)
    
    # quaternion normalisation loss
    _norm_loss = motion_vae_norm_loss(yhat)
    
    # position rec loss
    _pos_loss = motion_vae_pos_loss(y, yhat)
    
    # rotation rec loss
    _quat_loss = motion_vae_quat_loss(y, yhat)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * motion_vae_norm_loss_scale
    _total_loss += _pos_loss * motion_vae_pos_loss_scale
    _total_loss += _quat_loss * motion_vae_quat_loss_scale
    _total_loss += _kld_loss * vae_beta
    
    return _total_loss, _norm_loss, _pos_loss, _quat_loss, _kld_loss

"""
Audio VAE Training and Test Step
"""

def audio_vae_train_step(y_norm):
    
    #print("y_norm s ", y_norm.shape)
    
    y_norm = y_norm.unsqueeze(1).to(device)
    
    #print("y_norm 2 s ", y_norm.shape)
    
    # use only the last mels for y_norm for the vae training
    audio_encoder_input = y_norm[:,:,:, -audio_vae_mel_count:]
    
    #print("audio_encoder_input s ", audio_encoder_input.shape)
    
    audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
    mu = audio_encoder_output_mu
    std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
    
    audio_encoder_output = audio_encoder.reparameterize(mu, std)
    
    #print("audio_encoder_output s ", audio_encoder_output.shape)
    
    audio_decoder_input = audio_encoder_output
    
    #print("audio_decoder_input s ", audio_decoder_input.shape)
    
    audio_decoder_output = audio_decoder(audio_decoder_input)
    
    #print("audio_decoder_output s ", audio_decoder_output.shape)
    
    # kld loss
    _kld_loss = variational_loss(mu, std)
    
    # mel rec loss
    _mel_rec_loss = audio_vae_mel_loss(audio_encoder_input.squeeze(1), audio_decoder_output.squeeze(1))
    
    # perc rec loss
    y_mels_norm = y_norm.squeeze(1)
    yhat_mels_norm = audio_decoder_output.squeeze(1)
    
    #print("y_mels_norm s ", y_mels_norm.shape)
    #print("yhat_mels_norm s ", yhat_mels_norm.shape)
    
    # denormalise mel
    y_mels = y_mels_norm * audio_std + audio_mean
    yhat_mels = yhat_mels_norm * audio_std + audio_mean
    
    #print("y_mels s ", y_mels.shape)
    #print("yhat_mels s ", yhat_mels.shape)
    
    # prepend the first y_mels to yhat_norm to create the full number of mels required by vocos
    yhat_mels = torch.cat([y_mels[:, :, :-audio_vae_mel_count], yhat_mels], dim=2)
    
    #print("yhat_mels 2 s ", yhat_mels.shape)
    
    # convert mels to waveforms
    y_wave = vocos.decode(y_mels)
    yhat_wave = vocos.decode(yhat_mels)
    
    #print("y_wave s ", y_wave.shape)
    #print("yhat_wave s ", yhat_wave.shape)
    
    # take only the last part of y_wave and yhat_wave that correspond to audio_vae_waveform_length
    y_wave = y_wave[:, -audio_vae_waveform_length:]
    yhat_wave = yhat_wave[:, -audio_vae_waveform_length:]
    
    #print("y_wave 2 s ", y_wave.shape)
    #print("yhat_wave 2 s ", yhat_wave.shape)
    
    y_wave = y_wave.unsqueeze(1)
    yhat_wave = yhat_wave.unsqueeze(1)
    
    #print("y_wave 3 s ", y_wave.shape)
    #print("yhat_wave 3 s ", yhat_wave.shape)
    
    _perc_rec_loss = audio_vae_perc_loss(y_wave, yhat_wave)
    
    _total_loss = 0.0
    _total_loss += _mel_rec_loss * audio_vae_rec_loss_scale * 0.5
    _total_loss += _perc_rec_loss * audio_vae_rec_loss_scale * 0.5
    _total_loss += _kld_loss * vae_beta
    
    # Backpropagation
    audio_vae_optimizer.zero_grad()
    _total_loss.backward()
    
    audio_vae_optimizer.step()
    
    return _total_loss, _mel_rec_loss, _perc_rec_loss, _kld_loss

"""
_losses = audio_vae_train_step(batch_audio)
"""

@torch.no_grad()
def audio_vae_test_step(y_norm):
    
    #print("y_norm s ", y_norm.shape)
    
    y_norm = y_norm.unsqueeze(1).to(device)
    
    #print("y_norm 2 s ", y_norm.shape)
    
    # use only the last mels for y_norm for the vae training
    audio_encoder_input = y_norm[:,:,:, -audio_vae_mel_count:]
    
    #print("audio_encoder_input s ", audio_encoder_input.shape)
    
    audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
    mu = audio_encoder_output_mu
    std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
    
    audio_encoder_output = audio_encoder.reparameterize(mu, std)
    
    #print("audio_encoder_output s ", audio_encoder_output.shape)
    
    audio_decoder_input = audio_encoder_output
    
    #print("audio_decoder_input s ", audio_decoder_input.shape)
    
    audio_decoder_output = audio_decoder(audio_decoder_input)
    
    #print("audio_decoder_output s ", audio_decoder_output.shape)
    
    # kld loss
    _kld_loss = variational_loss(mu, std)
    
    # mel rec loss
    _mel_rec_loss = audio_vae_mel_loss(audio_encoder_input.squeeze(1), audio_decoder_output.squeeze(1))
    
    # perc rec loss
    y_mels_norm = y_norm.squeeze(1)
    yhat_mels_norm = audio_decoder_output.squeeze(1)
    
    #print("y_mels_norm s ", y_mels_norm.shape)
    #print("yhat_mels_norm s ", yhat_mels_norm.shape)
    
    # denormalise mel
    y_mels = y_mels_norm * audio_std + audio_mean
    yhat_mels = yhat_mels_norm * audio_std + audio_mean
    
    #print("y_mels s ", y_mels.shape)
    #print("yhat_mels s ", yhat_mels.shape)
    
    # prepend the first y_mels to yhat_norm to create the full number of mels required by vocos
    yhat_mels = torch.cat([y_mels[:, :, :-audio_vae_mel_count], yhat_mels], dim=2)
    
    #print("yhat_mels 2 s ", yhat_mels.shape)
    
    # convert mels to waveforms
    y_wave = vocos.decode(y_mels)
    yhat_wave = vocos.decode(yhat_mels)
    
    #print("y_wave s ", y_wave.shape)
    #print("yhat_wave s ", yhat_wave.shape)
    
    # take only the last part of y_wave and yhat_wave that correspond to audio_vae_waveform_length
    y_wave = y_wave[:, -audio_vae_waveform_length:]
    yhat_wave = yhat_wave[:, -audio_vae_waveform_length:]
    
    #print("y_wave 2 s ", y_wave.shape)
    #print("yhat_wave 2 s ", yhat_wave.shape)
    
    y_wave = y_wave.unsqueeze(1)
    yhat_wave = yhat_wave.unsqueeze(1)
    
    #print("y_wave 3 s ", y_wave.shape)
    #print("yhat_wave 3 s ", yhat_wave.shape)
    
    _perc_rec_loss = audio_vae_perc_loss(y_wave, yhat_wave)
    
    _total_loss = 0.0
    _total_loss += _mel_rec_loss * audio_vae_rec_loss_scale * 0.5
    _total_loss += _perc_rec_loss * audio_vae_rec_loss_scale * 0.5
    _total_loss += _kld_loss * vae_beta

    return _total_loss, _mel_rec_loss, _perc_rec_loss, _kld_loss

def contrastive_train_step(y_motion_norm, y_mels_norm):
    
    #print("y_motion_norm s ", y_motion_norm.shape)
    #print("y_mels_norm s ", y_mels_norm.shape)
    
    y_mels_norm = y_mels_norm.unsqueeze(1)
    #print("y_mels_norm 2 s ", y_mels_norm.shape)
    
    audio_encoder_input = y_mels_norm[:, :, :, -audio_vae_mel_count:]

    #print("audio_encoder_input s ", audio_encoder_input.shape)
    
    motion_encoder_input = y_motion_norm
    
    #print("motion_encoder_input s ", motion_encoder_input.shape)

    # Get embeddings from both encoders
    motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(motion_encoder_input)
    motion_mu = motion_encoder_output_mu
    motion_std = torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8
    motion_emb = motion_encoder.reparameterize(motion_mu, motion_std)
    
    #print("motion_emb s ", motion_emb.shape)
    
    audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
    audio_mu = audio_encoder_output_mu
    audio_std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
    audio_emb = audio_encoder.reparameterize(audio_mu, audio_std)
    
    #print("audio_emb s ", audio_emb.shape)
    
    # Compute contrastive loss
    loss = contrastive_loss_fn(motion_emb, audio_emb)
    
    # Backpropagation
    contrastive_optimizer.zero_grad()
    loss.backward()
    contrastive_optimizer.step()
    
    return loss
    
"""
_losses = contrastive_train_step(batch_mocap.to(device), batch_audio.to(device))
"""

@torch.no_grad()
def contrastive_test_step(y_motion_norm, y_mels_norm):
    
    #print("y_motion_norm s ", y_motion_norm.shape)
    #print("y_mels_norm s ", y_mels_norm.shape)
    
    y_mels_norm = y_mels_norm.unsqueeze(1)
    #print("y_mels_norm 2 s ", y_mels_norm.shape)
    
    audio_encoder_input = y_mels_norm[:, :, :, -audio_vae_mel_count:]

    #print("audio_encoder_input s ", audio_encoder_input.shape)
    
    motion_encoder_input = y_motion_norm
    
    #print("motion_encoder_input s ", motion_encoder_input.shape)

    # Get embeddings from both encoders
    motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(motion_encoder_input)
    motion_mu = motion_encoder_output_mu
    motion_std = torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8
    motion_emb = motion_encoder.reparameterize(motion_mu, motion_std)
    
    #print("motion_emb s ", motion_emb.shape)
    
    audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
    audio_mu = audio_encoder_output_mu
    audio_std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
    audio_emb = audio_encoder.reparameterize(audio_mu, audio_std)
    
    #print("audio_emb s ", audio_emb.shape)
    
    # Compute contrastive loss
    loss = contrastive_loss_fn(motion_emb, audio_emb)
    
    return loss

"""
Train Function
"""

def train(train_dataloader, test_dataloader, epochs):
    
    global vae_beta
    
    loss_history = {}
    loss_history["motion vae train"] = []
    loss_history["motion vae test"] = []
    loss_history["motion vae norm"] = []
    loss_history["motion vae pos"] = []
    loss_history["motion vae rot"] = []
    loss_history["motion vae kld"] = []
    
    loss_history["audio vae train"] = []
    loss_history["audio vae test"] = []
    loss_history["audio vae mel"] = []
    loss_history["audio vae perc"] = []
    loss_history["audio vae kld"] = []
    
    loss_history["vae contrastive train"] = []
    loss_history["vae contrastive test"] = []

    for epoch in range(epochs):

        start = time.time()
        
        vae_beta = vae_beta_values[epoch]
        
        #print("vae_beta ", vae_beta)
        
        motion_vae_train_loss_per_epoch = []
        motion_vae_test_loss_per_epoch = []
        motion_vae_norm_loss_per_epoch = []
        motion_vae_pos_loss_per_epoch = []
        motion_vae_rot_loss_per_epoch = []
        motion_vae_kld_loss_per_epoch = []
        
        audio_vae_train_loss_per_epoch = []
        audio_vae_test_loss_per_epoch = []
        audio_vae_mel_loss_per_epoch = []
        audio_vae_perc_loss_per_epoch = []
        audio_vae_kld_loss_per_epoch = []

        vae_contrastive_train_loss_per_epoch = []
        vae_contrastive_test_loss_per_epoch = []
        
        for motion_batch, audio_batch in train_dataloader:
            motion_batch = motion_batch.to(device)
            audio_batch = audio_batch.to(device)
            
            # motion vae
            _motion_vae_total_loss, _motion_vae_norm_loss, _motion_vae_pos_loss, _motion_vae_rot_loss, _motion_vae_kld_loss = motion_vae_train_step(motion_batch)
            
            _motion_vae_total_loss = _motion_vae_total_loss.detach().cpu().numpy()
            _motion_vae_norm_loss = _motion_vae_norm_loss.detach().cpu().numpy()
            _motion_vae_pos_loss = _motion_vae_pos_loss.detach().cpu().numpy()
            _motion_vae_rot_loss = _motion_vae_rot_loss.detach().cpu().numpy()
            _motion_vae_kld_loss = _motion_vae_kld_loss.detach().cpu().numpy()
            
            motion_vae_train_loss_per_epoch.append(_motion_vae_total_loss)
            motion_vae_norm_loss_per_epoch.append(_motion_vae_norm_loss)
            motion_vae_pos_loss_per_epoch.append(_motion_vae_pos_loss)
            motion_vae_rot_loss_per_epoch.append(_motion_vae_rot_loss)
            motion_vae_kld_loss_per_epoch.append(_motion_vae_kld_loss)
            
            # audio vae
            _audio_vae_total_loss, _audio_vae_mel_rec_loss, _audio_vae_perc_rec_loss, _audio_vae_kld_loss = audio_vae_train_step(audio_batch)
           
            _audio_vae_total_loss = _audio_vae_total_loss.detach().cpu().numpy()
            _audio_vae_mel_rec_loss = _audio_vae_mel_rec_loss.detach().cpu().numpy()
            _audio_vae_perc_rec_loss = _audio_vae_perc_rec_loss.detach().cpu().numpy()
            _audio_vae_kld_loss = _audio_vae_kld_loss.detach().cpu().numpy()
            
            audio_vae_train_loss_per_epoch.append(_audio_vae_total_loss)
            audio_vae_mel_loss_per_epoch.append(_audio_vae_mel_rec_loss)
            audio_vae_perc_loss_per_epoch.append(_audio_vae_perc_rec_loss)
            audio_vae_kld_loss_per_epoch.append(_audio_vae_kld_loss)
            
            # contrastive
            _vae_contrastive_loss = contrastive_train_step(motion_batch, audio_batch)
            _vae_contrastive_loss = _vae_contrastive_loss.detach().cpu().numpy()
            
            vae_contrastive_train_loss_per_epoch.append(_vae_contrastive_loss)

        motion_vae_train_loss_per_epoch = np.mean(np.array(motion_vae_train_loss_per_epoch))
        motion_vae_norm_loss_per_epoch = np.mean(np.array(motion_vae_norm_loss_per_epoch))
        motion_vae_pos_loss_per_epoch = np.mean(np.array(motion_vae_pos_loss_per_epoch))
        motion_vae_rot_loss_per_epoch = np.mean(np.array(motion_vae_rot_loss_per_epoch))
        motion_vae_kld_loss_per_epoch = np.mean(np.array(motion_vae_kld_loss_per_epoch))
        audio_vae_train_loss_per_epoch = np.mean(np.array(audio_vae_train_loss_per_epoch))
        audio_vae_mel_loss_per_epoch = np.mean(np.array(audio_vae_mel_loss_per_epoch))
        audio_vae_perc_loss_per_epoch = np.mean(np.array(audio_vae_perc_loss_per_epoch))
        audio_vae_kld_loss_per_epoch = np.mean(np.array(audio_vae_kld_loss_per_epoch))
        vae_contrastive_train_loss_per_epoch = np.mean(np.array(vae_contrastive_train_loss_per_epoch))

        for motion_batch, audio_batch in test_dataloader:
            motion_batch = motion_batch.to(device)
            audio_batch = audio_batch.to(device)
            
            # motion vae
            _motion_vae_total_loss, _, _, _, _ = motion_vae_test_step(motion_batch)
            
            _motion_vae_total_loss = _motion_vae_total_loss.detach().cpu().numpy()
            
            motion_vae_test_loss_per_epoch.append(_motion_vae_total_loss)

            # audio vae
            _audio_vae_total_loss, _, _, _ = audio_vae_test_step(audio_batch)
           
            _audio_vae_total_loss = _audio_vae_total_loss.detach().cpu().numpy()
            
            audio_vae_test_loss_per_epoch.append(_audio_vae_total_loss)
            
            # contrastive
            _vae_contrastive_loss = contrastive_test_step(motion_batch, audio_batch)
            _vae_contrastive_loss = _vae_contrastive_loss.detach().cpu().numpy()
            
            vae_contrastive_test_loss_per_epoch.append(_vae_contrastive_loss)
            
        motion_vae_test_loss_per_epoch = np.mean(np.array(motion_vae_test_loss_per_epoch))
        audio_vae_test_loss_per_epoch = np.mean(np.array(audio_vae_test_loss_per_epoch))
        vae_contrastive_test_loss_per_epoch = np.mean(np.array(vae_contrastive_test_loss_per_epoch))

        if epoch % model_save_interval == 0 and save_motion_vae_weights == True:
            torch.save(motion_encoder.state_dict(), "results/weights/motion_encoder_weights_epoch_{}".format(epoch))
            torch.save(motion_decoder.state_dict(), "results/weights/motion_decoder_weights_epoch_{}".format(epoch))

        if epoch % model_save_interval == 0 and save_audio_vae_weights == True:
            torch.save(audio_encoder.state_dict(), "results/weights/audio_encoder_weights_epoch_{}".format(epoch))
            torch.save(audio_decoder.state_dict(), "results/weights/audio_decoder_weights_epoch_{}".format(epoch))

        loss_history["motion vae train"].append(motion_vae_train_loss_per_epoch)
        loss_history["motion vae test"].append(motion_vae_test_loss_per_epoch)
        loss_history["motion vae norm"].append(motion_vae_norm_loss_per_epoch)
        loss_history["motion vae pos"].append(motion_vae_pos_loss_per_epoch)
        loss_history["motion vae rot"].append(motion_vae_rot_loss_per_epoch)
        loss_history["motion vae kld"].append(motion_vae_kld_loss_per_epoch)
        
        loss_history["audio vae train"].append(audio_vae_train_loss_per_epoch)
        loss_history["audio vae test"].append(audio_vae_test_loss_per_epoch)
        loss_history["audio vae mel"].append(audio_vae_mel_loss_per_epoch)
        loss_history["audio vae perc"].append(audio_vae_perc_loss_per_epoch)
        loss_history["audio vae kld"].append(audio_vae_kld_loss_per_epoch)
        
        loss_history["vae contrastive train"].append(vae_contrastive_train_loss_per_epoch)
        loss_history["vae contrastive test"].append(vae_contrastive_test_loss_per_epoch)
        
        print ("epoch {} : ".format(epoch + 1) +
               "motion train : {:01.4f} ".format(motion_vae_train_loss_per_epoch) +
               "motion test: {:01.4f} ".format(motion_vae_test_loss_per_epoch) +
               "motion norm: {:01.4f} ".format(motion_vae_norm_loss_per_epoch) + 
               "motion pos: {:01.4f} ".format(motion_vae_pos_loss_per_epoch) +
               "motion rot: {:01.4f} ".format(motion_vae_rot_loss_per_epoch) +
               "motion kld: {:01.4f} ".format(motion_vae_kld_loss_per_epoch) +
               "audio train : {:01.4f} ".format(audio_vae_train_loss_per_epoch) +
               "audio test : {:01.4f} ".format(audio_vae_test_loss_per_epoch) +
               "audio mel : {:01.4f} ".format(audio_vae_mel_loss_per_epoch) +
               "audio perc : {:01.4f} ".format(audio_vae_perc_loss_per_epoch) +
               "audio kld : {:01.4f} ".format(audio_vae_kld_loss_per_epoch) +
               "contr train : {:01.4f} ".format(vae_contrastive_train_loss_per_epoch) +
               "contr test : {:01.4f} ".format(vae_contrastive_test_loss_per_epoch) +
               "time {:01.2f} ".format(time.time()-start))

        motion_vae_scheduler.step()
        audio_vae_scheduler.step()
        contrastive_scheduler.step()
        
    return loss_history
"""
Fit Model
"""

loss_history = train(train_loader, test_loader, epochs)

"""
Save Training History
"""

def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.show()
    
def save_loss_as_image_normalised(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        losses = np.array(loss_history[key])
        loss_norm = losses / np.max(losses)
        plt.plot(range(epochs), loss_norm, label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.show()
    

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)


save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))
save_loss_as_image_normalised(loss_history, "results/histories/history_norm_{}.png".format(epochs))

"""
Save Final Model Weights
"""

if save_motion_vae_weights == True:
    torch.save(motion_encoder.state_dict(), "results/weights/motion_encoder_weights_epoch_{}".format(epochs))
    torch.save(motion_decoder.state_dict(), "results/weights/motion_decoder_weights_epoch_{}".format(epochs))

if save_audio_vae_weights == True:
    torch.save(audio_encoder.state_dict(), "results/weights/audio_encoder_weights_epoch_{}".format(epochs))
    torch.save(audio_decoder.state_dict(), "results/weights/audio_decoder_weights_epoch_{}".format(epochs))

"""
Inference
"""

import colorsys
from matplotlib.colors import hsv_to_rgb

def distinct_hsv_colors(n, s=0.8, v=0.9):
    """
    Generate n HSV colors (h, s, v in [0, 1]) with
    evenly spaced hues and fixed saturation/value.
    """
    if n <= 0:
        return []
    # Golden ratio conjugate gives decent spacing on [0, 1)
    phi = 0.618033988749895
    colors = []
    h = 0.0
    for _ in range(n):
        colors.append((h % 1.0, s, v))
        h += phi
    return colors

"""
Motion Inference
"""

poseRenderer = PoseRenderer(edge_list)

def export_motion_anim(pose_sequence, file_name):

    pose_count = pose_sequence.shape[0]

    pose_sequence = np.reshape(pose_sequence, (pose_count, mocap_joint_count, mocap_joint_dim))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)

    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)   

    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_motion_bvh(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]

    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_local"] = pose_sequence
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])

    pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
    
    bvh_tools.write(pred_bvh, file_name)

def export_motion_fbx(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    
    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_local"] = pose_sequence
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])
    
    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    
    fbx_tools.write(pred_fbx, file_name)

@torch.no_grad()
def encode_motion(orig_sequence, frame_indices):
    
    motion_encoder.eval()
    
    latent_vectors = []
    
    seq_excerpt_count = len(frame_indices)

    for excerpt_index in range(seq_excerpt_count):
        excerpt_start_frame = frame_indices[excerpt_index]
        excerpt_end_frame = excerpt_start_frame + motion_vae_mocap_length

        excerpt = orig_sequence[excerpt_start_frame:excerpt_end_frame]
        excerpt = np.expand_dims(excerpt, axis=0)
        excerpt = torch.from_numpy(excerpt).reshape(1, motion_vae_mocap_length, mocap_pose_dim).to(device)

        excerpt_norm = (excerpt - mocap_mean) / mocap_std + 1e-8

        encoder_output_mu, encoder_output_std = motion_encoder(excerpt_norm)
        mu = encoder_output_mu
        std = torch.nn.functional.softplus(encoder_output_std) + 1e-8
        latent_vector = motion_encoder.reparameterize(mu, std)
            
        latent_vector = torch.squeeze(latent_vector)
        latent_vector = latent_vector.detach().cpu().numpy()

        latent_vectors.append(latent_vector)
        
    motion_encoder.train()
        
    return latent_vectors

@torch.no_grad()
def decode_motion(sequence_encodings, seq_offset, base_pose):
    
    motion_decoder.eval()
    
    seq_env = np.hanning(motion_vae_mocap_length)
    seq_excerpt_count = len(sequence_encodings)
    gen_seq_length = (seq_excerpt_count - 1) * seq_offset + motion_vae_mocap_length

    gen_sequence = np.full(shape=(gen_seq_length, mocap_joint_count, mocap_joint_dim), fill_value=base_pose)
    
    for excerpt_index in range(len(sequence_encodings)):
        latent_vector = sequence_encodings[excerpt_index]
        latent_vector = np.expand_dims(latent_vector, axis=0)
        latent_vector = torch.from_numpy(latent_vector).to(device)
        
        excerpt_norm = motion_decoder(latent_vector)
        
        excerpt_dec = excerpt_norm * mocap_std + mocap_mean
        
        excerpt_dec = torch.squeeze(excerpt_dec)
        excerpt_dec = excerpt_dec.detach().cpu().numpy()
        excerpt_dec = np.reshape(excerpt_dec, (-1, mocap_joint_count, mocap_joint_dim))
        
        gen_frame = excerpt_index * seq_offset
        
        for si in range(motion_vae_mocap_length):
            for ji in range(mocap_joint_count): 
                current_quat = gen_sequence[gen_frame + si, ji, :]
                target_quat = excerpt_dec[si, ji, :]
                quat_mix = seq_env[si]
                mix_quat = slerp(current_quat, target_quat, quat_mix )
                gen_sequence[gen_frame + si, ji, :] = mix_quat
        
    gen_sequence = gen_sequence.reshape((-1, 4))
    gen_sequence = gen_sequence / np.linalg.norm(gen_sequence, ord=2, axis=1, keepdims=True)
    gen_sequence = gen_sequence.reshape((gen_seq_length, mocap_joint_count, mocap_joint_dim))
    gen_sequence = qfix(gen_sequence)

    motion_decoder.train()
    
    return gen_sequence

@torch.no_grad()
def create_motion_space_representation(sequence_excerpts):
    
    motion_encoder.eval()

    encodings = []
    
    excerpt_count = sequence_excerpts.shape[0]
    
    for eI in range(0, excerpt_count, batch_size):
        
        excerpt = sequence_excerpts[eI:eI+batch_size]
 
        excerpt = torch.from_numpy(excerpt).to(device)
        
        encoder_output_mu, encoder_output_std = motion_encoder(excerpt)
        mu = encoder_output_mu
        std = torch.nn.functional.softplus(encoder_output_std) + 1e-8
        latent_vector = motion_encoder.reparameterize(mu, std)

        latent_vector = latent_vector.detach().cpu()

        encodings.append(latent_vector)
        
    encodings = torch.cat(encodings, dim=0)
    
    #print("encodings s ", encodings.shape)
    
    encodings = encodings.numpy()

    # use TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, n_iter=5000, verbose=1)    
    Z_tsne = tsne.fit_transform(encodings)
    
    motion_encoder.train()
    
    return Z_tsne

def create_motion_space_image(Z_tsne, motion_highlight_excerpt_ranges, file_name):
    
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]
    
    plot_colors_hsv = distinct_hsv_colors(len(motion_highlight_excerpt_ranges))
    plot_colors = [ hsv_to_rgb(hsv)  for hsv in plot_colors_hsv ]
    
    plt.figure()
    fig, ax = plt.subplots()
    #ax.plot(Z_tsne_x, Z_tsne_y, '-', c="grey",linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.5)

    for hI, hR in enumerate(motion_highlight_excerpt_ranges):
        #ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], marker='x', facecolors=plot_colors[hI], s=5.0, linewidths=0.2, edgecolors=plot_colors[hI], alpha=0.4)
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=600)
    plt.close()
    
"""
Create Motion Space Plot
"""

motion_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0]
motion_highlight_duration_sec = 10.0

motion_highlight_ranges = []

for motion_highlight_start_sec in motion_highlight_starts_sec:
    motion_highlight_ranges.append([ int(motion_highlight_start_sec * mocap_fps) // mocap_offset, int((motion_highlight_start_sec + motion_highlight_duration_sec) * mocap_fps) // mocap_offset ])

Z_tsne = create_motion_space_representation(mocap_dataset.numpy())
create_motion_space_image(Z_tsne, motion_highlight_ranges, "motion_space_plot_epoch_{}.png".format(epochs))

"""
Create Original Motion
"""

mocap_data = mocap_all_data[0]
mocap_motion_orig = mocap_data["motion"]["rot_local"].astype(np.float32)

seq_start = 7000
seq_length = 1000

export_motion_anim(mocap_motion_orig[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.gif".format(seq_start, seq_length))
export_motion_fbx(mocap_motion_orig[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.fbx".format(seq_start, seq_length))

"""
Recontruct Original Motion
"""

motion_start_sec = 130.0
motion_duration_sec = 10.0
motion_offset = motion_vae_mocap_length // 4 
base_pose = np.reshape(mocap_motion_orig[0], (mocap_joint_count, mocap_joint_dim))

motion_indices = [ frame_index for frame_index in range(int(motion_start_sec * mocap_fps), int((motion_start_sec + motion_duration_sec) * mocap_fps), motion_offset)]

motion_encodings = encode_motion(mocap_motion_orig, motion_indices)
gen_motion = decode_motion(motion_encodings, motion_offset, base_pose)
export_motion_anim(gen_motion, "results/anims/rec_motion_start_{}_length_{}_epoch_{}.gif".format(motion_start_sec, motion_duration_sec, epochs))
export_motion_fbx(gen_motion, "results/anims/rec_motion_start_{}_length_{}_epoch_{}.fbx".format(motion_start_sec, motion_duration_sec, epochs))

"""
Random Walk in Motion Space
"""

motion_start_sec = 130.0
motion_duration_sec = 10.0

motion_indices = [ int(motion_start_sec * mocap_fps) ]

motion_encodings = encode_motion(mocap_motion_orig, motion_indices)

for index in range(0, int((motion_duration_sec) * mocap_fps)  // motion_offset):
    random_step = np.random.random((motion_vae_latent_dim)).astype(np.float32) * 2.0
    motion_encodings.append(motion_encodings[index] + random_step)
    
gen_motion = decode_motion(motion_encodings, motion_offset, base_pose)
export_motion_anim(gen_motion, "results/anims/randwalk_motion_start_{}_length_{}_epoch_{}.gif".format(motion_start_sec, motion_duration_sec, epochs))
export_motion_fbx(gen_motion, "results/anims/randwalk_motion_start_{}_length_{}_epoch_{}.fbx".format(motion_start_sec, motion_duration_sec, epochs))

"""
Sequence Offset Following in Motion Space
"""

motion_start_sec = 130.0
motion_duration_sec = 10.0
    
motion_indices = [ seq_index for seq_index in range(int(motion_start_sec * mocap_fps), int((motion_start_sec + motion_duration_sec) * mocap_fps), motion_offset)]

motion_encodings = encode_motion(mocap_motion_orig, motion_indices)

offset_motion_encodings = []

for index in range(len(motion_encodings)):
    sin_value = np.sin(index / (len(motion_encodings) - 1) * np.pi * 4.0)
    offset = np.ones(shape=(motion_vae_latent_dim), dtype=np.float32) * sin_value * 4.0
    offset_motion_encoding = motion_encodings[index] + offset
    offset_motion_encodings.append(offset_motion_encoding)
    
gen_motion = decode_motion(offset_motion_encodings, motion_offset, base_pose)
export_motion_anim(gen_motion, "results/anims/offset_motion_start_{}_length_{}_epoch_{}.gif".format(motion_start_sec, motion_duration_sec, epochs))
export_motion_fbx(gen_motion, "results/anims/offset_motion_start_{}_length_{}_epoch_{}.fbx".format(motion_start_sec, motion_duration_sec, epochs))

"""
Interpolate Two Motion Excerpts in Motion Space
"""

motion1_start_sec = 130.0
motion2_start_sec = 240.0
motion_duration_sec = 10.0

motion1_indices = [ seq_index for seq_index in range(int(motion1_start_sec * mocap_fps), int((motion1_start_sec + motion_duration_sec) * mocap_fps), motion_offset)]
motion2_indices = [ seq_index for seq_index in range(int(motion2_start_sec * mocap_fps), int((motion2_start_sec + motion_duration_sec) * mocap_fps), motion_offset)]

motion1_encodings = encode_motion(mocap_motion_orig, motion1_indices)
motion2_encodings = encode_motion(mocap_motion_orig, motion2_indices)

mix_motion_encodings = []

for index in range(len(motion1_encodings)):
    mix_factor = index / (len(motion1_encodings) - 1)
    mix_motion_encoding = motion1_encodings[index] * (1.0 - mix_factor) + motion2_encodings[index] * mix_factor
    mix_motion_encodings.append(mix_motion_encoding)
    
gen_motion = decode_motion(mix_motion_encodings, motion_offset, base_pose)
export_motion_anim(gen_motion, "results/anims/mix_motion_start1_{}_start2_{}_length_{}_epoch_{}.gif".format(motion1_start_sec, motion2_start_sec, motion_duration_sec, epochs))
export_motion_fbx(gen_motion, "results/anims/mix_motion_start1_{}_start2_{}_length_{}_epoch_{}.fbx".format(motion1_start_sec, motion2_start_sec, motion_duration_sec, epochs))


"""
Audio Inference
"""

def export_audio(waveform, file_name):
    
    torchaudio.save("{}".format(file_name), waveform, audio_sample_rate)
    
def export_vocos_audio(waveform, file_name):
    
    waveform_length = waveform.shape[1]
    audio_window_offset = audio_vocos_waveform_length // 2
    audio_window_env = torch.hann_window(audio_vocos_waveform_length)
    
    audio_window_count = int(waveform_length - audio_vocos_waveform_length) // audio_window_offset
    pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
    
    #print("pred_audio_sequence s ", pred_audio_sequence.shape)
    
    for i in range(audio_window_count):
        
        window_start = i * audio_window_offset
        window_end = window_start + audio_vocos_waveform_length
        
        target_audio = waveform[:, window_start:window_end]
        
        #print("i ", i, " target_audio s ", target_audio.shape)
        
        with torch.no_grad():
            audio_features = vocos.feature_extractor(target_audio.to(device))
            voc_audio = vocos.decode(audio_features).detach().cpu()

        #print("voc_audio s ", voc_audio.shape)
        #print("grain_env s ", grain_env.shape)

        pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_vocos_waveform_length] += voc_audio[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_sample_rate)

@torch.no_grad()
def encode_audio(orig_waveform, sample_indices):
    
    audio_encoder.eval()
    
    waveform = orig_waveform.to(device)
    waveform_length = waveform.shape[-1]
    
    latent_vectors = []
    
    waveform_excerpt_count = len(sample_indices)
    
    for excerpt_index in range(waveform_excerpt_count):
        
        excerpt_start_sample = sample_indices[excerpt_index]
        excerpt_end_sample = excerpt_start_sample + audio_vocos_waveform_length
        
        audio_waveform_excerpt = waveform[:, excerpt_start_sample:excerpt_end_sample].to(device)
        
        #print("audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
        
        audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
        
        #print("audio_mels_excerpt s ", audio_mels_excerpt.shape)
        
        # normalise audio mels excerpt
        audio_mels_excerpt_norm = (audio_mels_excerpt - audio_mean) / (audio_std + 1e-8)
        
        #print("audio_mels_excerpt_norm s ", audio_mels_excerpt_norm.shape)
        
        audio_mels_excerpt_norm = audio_mels_excerpt_norm.unsqueeze(1)
        
        # use only the last audio mels
        audio_encoder_input = audio_mels_excerpt_norm[:,:,:, -audio_vae_mel_count:]
        
        #print("audio_encoder_input s ", audio_encoder_input.shape)
        
        # excode audio mels
        audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
        mu = audio_encoder_output_mu
        std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
        
        latent_vector = audio_encoder.reparameterize(mu, std)
        
        #print("latent_vector s ", latent_vector.shape)
        
        latent_vector = torch.squeeze(latent_vector)
        latent_vector = latent_vector.detach().cpu().numpy()

        latent_vectors.append(latent_vector)
    
    audio_encoder.train()
    
    return latent_vectors

@torch.no_grad()
def decode_audio(latent_vectors, waveform_offset):
    
    audio_decoder.eval()
    
    seq_env = np.hanning(audio_vae_waveform_length)
    latent_count = len(latent_vectors)
    waveform_length = (latent_count - 1) * waveform_offset + audio_vae_waveform_length

    vocos_mels = vocos.feature_extractor(torch.zeros((1, audio_vocos_waveform_length), dtype=torch.float32).to(device))

    #print("vocos_mels s ", vocos_mels.shape)

    gen_waveform = np.zeros(shape=(waveform_length)).astype(np.float32)
    
    for latent_index in range(len(latent_vectors)):
        
        latent_vector = latent_vectors[latent_index]
        latent_vector = np.expand_dims(latent_vector, axis=0)
        latent_vector = torch.from_numpy(latent_vector).to(device)
        
        #print("latent_vector s ", latent_vector.shape)
        
        vae_mels_norm = audio_decoder(latent_vector)
        
        #print("vae_mels_norm s ", vae_mels_norm.shape)
        
        vae_mels_norm = vae_mels_norm.squeeze(1)
        
        #print("vae_mels_norm 2 s ", vae_mels_norm.shape)
        
        vae_mels = vae_mels_norm * audio_std + audio_mean
        
        #print("vae_mels s ", vae_mels.shape)
        
        vocos_mels = torch.roll(vocos_mels, shifts=-audio_vae_mel_count, dims=2)
        vocos_mels[:, :, -audio_vae_mel_count:] = vae_mels
        
        vocos_waveform = vocos.decode(vocos_mels)
        
        #print("vocos_waveform s ", vocos_waveform.shape)
        
        vae_waveform = vocos_waveform[0, -audio_vae_waveform_length:]
        
        #print("vae_waveform s ", vae_waveform.shape)
        
        vae_waveform = vae_waveform.detach().detach().cpu().numpy()
        
        asI = latent_index * waveform_offset
        gen_waveform[asI:asI + audio_vae_waveform_length] += vae_waveform * seq_env
    
    return np.reshape(gen_waveform, (1, -1))

@torch.no_grad()
def create_audio_space_representation(sequence_excerpts):
    
    audio_encoder.eval()

    encodings = []
    
    excerpt_count = sequence_excerpts.shape[0]
    
    for eI in range(0, excerpt_count, batch_size):
        
        excerpt = sequence_excerpts[eI:eI+batch_size]
 
        excerpt = torch.from_numpy(excerpt).to(device)
        
        excerpt = excerpt.unsqueeze(1)

        # use only the last mels
        excerpt = excerpt[:,:,:, -audio_vae_mel_count:]
  
        encoder_output_mu, encoder_output_std = audio_encoder(excerpt)
        mu = encoder_output_mu
        std = torch.nn.functional.softplus(encoder_output_std) + 1e-8
        latent_vector = audio_encoder.reparameterize(mu, std)

        latent_vector = latent_vector.detach().cpu()

        encodings.append(latent_vector)
        
    encodings = torch.cat(encodings, dim=0)
    
    #print("encodings s ", encodings.shape)
    
    encodings = encodings.numpy()

    # use TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, n_iter=5000, verbose=1)    
    Z_tsne = tsne.fit_transform(encodings)
    
    audio_encoder.train()
    
    return Z_tsne


def create_audio_space_image(Z_tsne, audio_highlight_excerpt_ranges, file_name):
    
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]
    
    plot_colors_hsv = distinct_hsv_colors(len(audio_highlight_excerpt_ranges))
    plot_colors = [ hsv_to_rgb(hsv)  for hsv in plot_colors_hsv ]
    
    plt.figure()
    fig, ax = plt.subplots()
    #ax.plot(Z_tsne_x, Z_tsne_y, '-', c="grey",linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.5)
        
    for hI, hR in enumerate(audio_highlight_excerpt_ranges):
        
        #ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], marker='o', facecolors="none", s=10.0, linewidths=0.2, edgecolors=plot_colors[hI], alpha=0.4)
        
        """
        print("begin >>>>> ", hI, " <<<<<<")
        
        print("audio hI ", hI, " color ", plot_colors[hI], " range ", hR[0], " - ", hR[1], " Z_tsne_x s ", Z_tsne_x.shape, " Z_tsne_y s ", Z_tsne_y.shape)

        print("Z_tsne_x[hR[0]:hR[1]] ", Z_tsne_x[hR[0]:hR[1]])
        print("Z_tsne_y[hR[0]:hR[1]] ", Z_tsne_y[hR[0]:hR[1]])
        
        print("end >>>>> ", hI, " <<<<<<")
        """
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=600)
    plt.close()
    
"""
Create Audio Space Plot
"""

audio_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0]
audio_highlight_duration_sec = 10.0

audio_highlight_ranges = []

for audio_highlight_start_sec in audio_highlight_starts_sec:
    audio_highlight_ranges.append([ int(audio_highlight_start_sec * audio_sample_rate) // audio_waveform_offset, int((audio_highlight_start_sec + audio_highlight_duration_sec) * audio_sample_rate) // audio_waveform_offset ])

Z_tsne = create_audio_space_representation(audio_dataset.numpy())
create_audio_space_image(Z_tsne, audio_highlight_ranges, "audio_space_plot_epoch_{}.png".format(epochs))

"""
Create Original Audio
"""

audio_data = audio_all_data[0]

audio_start_sec = 130.0
audio_duration_sec = 10.0

export_audio(audio_data[:, int(audio_start_sec * audio_sample_rate):int(audio_start_sec* audio_sample_rate + audio_duration_sec* audio_sample_rate)], "results/audio/orig_audio_start_{}_length_{}.wav".format(audio_start_sec, audio_duration_sec))
export_vocos_audio(audio_data[:, int(audio_start_sec * audio_sample_rate):int(audio_start_sec* audio_sample_rate + audio_duration_sec* audio_sample_rate)], "results/audio/orig_vocos_audio_start_{}_length_{}.wav".format(audio_start_sec, audio_duration_sec))

"""
Recontruct Original Audio
"""

audio_start_sec = 130.0
audio_duration_sec = 10.0
audio_waveform_offset = audio_vae_waveform_length // 4

sample_indices = [ sample_index for sample_index in range(int(audio_start_sec * audio_sample_rate), int((audio_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]

audio_encodings = encode_audio(audio_data, sample_indices)
gen_audio = decode_audio(audio_encodings, audio_waveform_offset)
export_audio(torch.from_numpy(gen_audio), "results/audio/rec_audio_start_{}_length_{}_epochs_{}.wav".format(audio_start_sec, audio_duration_sec, epochs))

"""
Random Walk in Audio Space
"""

audio_start_sec = 130.0
audio_duration_sec = 10.0

sample_indices = [int(audio_start_sec * audio_sample_rate)]

audio_encodings = encode_audio(audio_data, sample_indices)

for index in range(0, int(audio_duration_sec * audio_sample_rate) // audio_waveform_offset):
    random_step = np.random.random((audio_vae_latent_dim)).astype(np.float32) * 2.0
    audio_encodings.append(audio_encodings[index] + random_step)
    
gen_audio = decode_audio(audio_encodings, audio_waveform_offset)
export_audio(torch.from_numpy(gen_audio), "results/audio/randwalk_audio_start_{}_length_{}_epochs_{}.wav".format(audio_start_sec, audio_duration_sec, epochs))

"""
Sequence Offset Following in Audio Space
"""

audio_start_sec = 130.0
audio_duration_sec = 10.0

sample_indices = [ sample_index for sample_index in range(int(audio_start_sec * audio_sample_rate), int((audio_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]

audio_encodings = encode_audio(audio_data, sample_indices)

offset_audio_encodings = []

for index in range(len(audio_encodings)):
    sin_value = np.sin(index / (len(audio_encodings) - 1) * np.pi * 4.0)
    offset = np.ones(shape=(audio_vae_latent_dim), dtype=np.float32) * sin_value * 4.0
    offset_audio_encoding = audio_encodings[index] + offset
    offset_audio_encodings.append(offset_audio_encoding)
    
gen_audio = decode_audio(offset_audio_encodings, audio_waveform_offset)
export_audio(torch.from_numpy(gen_audio), "results/audio/offset_audio_start_{}_length_{}_epochs_{}.wav".format(audio_start_sec, audio_duration_sec, epochs))

"""
Interpolate Two Audio Excerpts in Audio Space
"""

audio1_start_sec = 130.0
audio2_start_sec = 240.0
audio_duration_sec = 10.0

sample1_indices = [ sample_index for sample_index in range(int(audio1_start_sec * audio_sample_rate), int((audio1_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]
sample2_indices = [ sample_index for sample_index in range(int(audio2_start_sec * audio_sample_rate), int((audio2_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]

audio1_encodings = encode_audio(audio_data, sample1_indices)
audio2_encodings = encode_audio(audio_data, sample2_indices)

mix_encodings = []

for index in range(len(audio1_encodings)):
    mix_factor = index / (len(audio1_encodings) - 1)
    mix_encoding = audio1_encodings[index] * (1.0 - mix_factor) + audio2_encodings[index] * mix_factor
    mix_encodings.append(mix_encoding)
    
gen_audio = decode_audio(mix_encodings, audio_waveform_offset)
export_audio(torch.from_numpy(gen_audio), "results/audio/mix_audio_start1_{}_start2_{}_length_{}_epochs_{}.wav".format(audio1_start_sec, audio2_start_sec, audio_duration_sec, epochs))

"""
Motion and Audio Inference
"""

@torch.no_grad()
def create_motion_audio_space_representation(motion_excerpts, audio_excerpts):
    
    motion_encoder.eval()
    audio_encoder.eval()

    motion_encodings = []
    audio_encodings = []
    
    motion_excerpt_count = motion_excerpts.shape[0]
    audio_excerpt_count = audio_excerpts.shape[0]
    
    for eI in range(0, motion_excerpt_count, batch_size):
        
        excerpt = motion_excerpts[eI:eI+batch_size]
 
        excerpt = torch.from_numpy(excerpt).to(device)
        
        encoder_output_mu, encoder_output_std = motion_encoder(excerpt)
        mu = encoder_output_mu
        std = torch.nn.functional.softplus(encoder_output_std) + 1e-8
        latent_vector = motion_encoder.reparameterize(mu, std)

        latent_vector = latent_vector.detach().cpu()

        motion_encodings.append(latent_vector)
        
    for eI in range(0, audio_excerpt_count, batch_size):
        
        excerpt = audio_excerpts[eI:eI+batch_size]
    
        excerpt = torch.from_numpy(excerpt).to(device)
        
        excerpt = excerpt.unsqueeze(1)
    
        # use only the last mels
        excerpt = excerpt[:,:,:, -audio_vae_mel_count:]
    
        encoder_output_mu, encoder_output_std = audio_encoder(excerpt)
        mu = encoder_output_mu
        std = torch.nn.functional.softplus(encoder_output_std) + 1e-8
        latent_vector = audio_encoder.reparameterize(mu, std)
    
        latent_vector = latent_vector.detach().cpu()
    
        audio_encodings.append(latent_vector)

    motion_encodings = torch.cat(motion_encodings, dim=0)
    audio_encodings = torch.cat(audio_encodings, dim=0)
    
    motion_audio_encodings = torch.cat([motion_encodings, audio_encodings], dim=0)
    
    motion_audio_encodings = motion_audio_encodings.numpy()

    # use TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, n_iter=5000, verbose=1)    
    Z_tsne = tsne.fit_transform(motion_audio_encodings)
    
    motion_encoder.train()
    audio_encoder.train()
    
    return Z_tsne

def create_motion_audio_space_image(Z_tsne, motion_highlight_excerpt_ranges, audio_highlight_excerpt_ranges, file_name):
    
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]
    
    plot_colors_hsv = distinct_hsv_colors(len(motion_highlight_excerpt_ranges))
    plot_colors = [ hsv_to_rgb(hsv)  for hsv in plot_colors_hsv ]
    
    plt.figure()
    fig, ax = plt.subplots()
    #ax.plot(Z_tsne_x, Z_tsne_y, '-', c="grey",linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.5)

    for hI, hR in enumerate(motion_highlight_excerpt_ranges):
        #ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], marker='x', facecolors=plot_colors[hI], s=5.0, linewidths=0.2, edgecolors=plot_colors[hI], alpha=0.4)
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')
        
    for hI, hR in enumerate(audio_highlight_excerpt_ranges):
        #ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], marker='o', facecolors="none", s=10.0, linewidths=0.2, edgecolors=plot_colors[hI], alpha=0.4)
        
        #ax.set_xlabel('$c_1$')
        #ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=600)
    plt.close()
    

"""
Create Motion Audio Space Plot
"""

Z_tsne = create_motion_audio_space_representation(mocap_dataset.numpy(), audio_dataset.numpy())

motion_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0 ]
motion_highlight_duration_sec = 10.0

motion_highlight_ranges = []

for motion_highlight_start_sec in motion_highlight_starts_sec:
    motion_highlight_ranges.append([ int(motion_highlight_start_sec * mocap_fps) // motion_offset, int((motion_highlight_start_sec + motion_highlight_duration_sec) * mocap_fps) // motion_offset ])


audio_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0 ]
audio_highlight_duration_sec = 10.0

audio_highlight_ranges = []

audio_range_offset = mocap_dataset.shape[0]

for audio_highlight_start_sec in audio_highlight_starts_sec:

    audio_highlight_ranges.append([ int(audio_highlight_start_sec * audio_sample_rate) // audio_waveform_offset + audio_range_offset, int((audio_highlight_start_sec + audio_highlight_duration_sec) * audio_sample_rate) // audio_waveform_offset + audio_range_offset])

create_motion_audio_space_image(Z_tsne, motion_highlight_ranges, audio_highlight_ranges, "motion_audio_space_plot_epoch_{}.png".format(epochs))

len(motion_highlight_ranges)
len(audio_highlight_ranges)

motion_tmp = [ motion_range[1] - motion_range[0] for motion_range in motion_highlight_ranges ]
audio_tmp = [ audio_range[1] - audio_range[0] for audio_range in audio_highlight_ranges ]

for i in range(len(motion_highlight_ranges)):
    print("i ", i, " motion count ", motion_tmp[i], " audio_count ", audio_tmp[i] )
    

"""
Motion to Audio Inference
"""

motion_start_sec = 130.0
motion_duration_sec = 10.0
motion_offset = motion_vae_mocap_length // 4 
audio_waveform_offset = audio_vae_waveform_length // 4

motion_indices = [ frame_index for frame_index in range(int(motion_start_sec * mocap_fps), int((motion_start_sec + motion_duration_sec) * mocap_fps), motion_offset)]

motion_encodings = encode_motion(mocap_motion_orig, motion_indices)
gen_audio = decode_audio(motion_encodings, audio_waveform_offset)

export_audio(torch.from_numpy(gen_audio), "results/audio/motion2audio_motion_start_{}_length_{}_epochs_{}.wav".format(motion_start_sec, motion_duration_sec, epochs))

"""
Audio to Motion Inference
"""

audio_start_sec = 130.0
audio_duration_sec = 10.0
audio_waveform_offset = audio_vae_waveform_length // 4
motion_offset = motion_vae_mocap_length // 4 

audio_indices = [ sample_index for sample_index in range(int(audio_start_sec * audio_sample_rate), int((audio_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]

audio_encodings = encode_audio(audio_data, sample_indices)
gen_motion = decode_motion(audio_encodings, motion_offset, base_pose)

export_motion_anim(gen_motion, "results/anims/audio2motion_audio_start_{}_length_{}_epochs_{}.wav.gif".format(audio_start_sec, audio_duration_sec, epochs))
export_motion_fbx(gen_motion, "results/anims/audio2motion_audio_start_{}_length_{}_epochs_{}.wav.fbx".format(audio_start_sec, audio_duration_sec, epochs))

