"""
same as motion_audio_transformer_vae_vocos_4_2.py 
but uses Multmodal VAE for motion and audio
"""

"""
Imports
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import scipy.linalg as sclinalg

import math
import os, sys, time, subprocess
import numpy as np
import csv
import matplotlib.pyplot as plt

# audio specific imports

import torchaudio
import torchaudio.transforms as transforms
import simpleaudio as sa
import auraloss

# vocos specific imports
from vocos import Vocos 

# mocap specific imports

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_data_path = "E:/Data/mocap/Eleni/Solos/ZHdK_04.12.2025/fbx_60hz"
mocap_data_files = ["Eline_Session-002.fbx"]
mocap_valid_ranges = [[780, 50163]]

mocap_pos_scale = 1.0
mocap_fps = 60

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

motion_vae_encoder_weights_file = "../../../VAE/Multimodal_VAE/Multimodal_VAE_Vocos_CL/Training/results_Eleni_Take2_mocap8_kld1.0_cl1.0/weights/motion_encoder_weights_epoch_400"
motion_vae_decoder_weights_file = "../../../VAE/Multimodal_VAE/Multimodal_VAE_Vocos_CL/Training/results_Eleni_Take2_mocap8_kld1.0_cl1.0/weights/motion_decoder_weights_epoch_400"

load_motion_latents_stats = False
motion_latents_mean_file = "results/stats/motion_latents_mean.pt"
motion_latents_std_file = "results/stats/motion_latents_std.pt"


"""
Audio VAE Model Settings
"""

audio_vae_waveform_length = int(motion_vae_mocap_length / mocap_fps * audio_sample_rate)
audio_vae_mel_count = None # automatically calculated
audio_waveform_start_offset = audio_vocos_waveform_length - audio_vae_waveform_length
mocap_frame_start_offset = int(audio_waveform_start_offset / audio_sample_rate * mocap_fps)

audio_vae_latent_dim = 128
audio_vae_conv_channel_counts = [ 16, 32, 64, 128 ]
audio_vae_conv_kernel_size = (5, 3)
audio_vae_dense_layer_sizes = [ 512 ]

audio_vae_encoder_weights_file = "../../../VAE/Multimodal_VAE/Multimodal_VAE_Vocos_CL/Training/results_Eleni_Take2_mocap8_kld1.0_cl1.0/weights/audio_encoder_weights_epoch_400"
audio_vae_decoder_weights_file = "../../../VAE/Multimodal_VAE/Multimodal_VAE_Vocos_CL/Training/results_Eleni_Take2_mocap8_kld1.0_cl1.0/weights/audio_decoder_weights_epoch_400"

load_audio_latents_stats = False
audio_latents_mean_file = "results/stats/audio_latents_mean.pt"
audio_latents_std_file = "results/stats/audio_latents_std.pt"

"""
Transformer Model Settings
"""

transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 256
transformer_dropout = 0.1   

transformer_mocap_input_length = 56
transformer_mocap_output_length = 10 
transformer_mocap_output_length_2 = transformer_mocap_output_length + motion_vae_mocap_length - 1

transformer_audio_waveform_input_length = int(transformer_mocap_input_length / mocap_fps * audio_sample_rate)
transformer_audio_waveform_output_length = int(transformer_mocap_output_length / mocap_fps * audio_sample_rate)
transformer_audio_waveform_output_length_2 = int(transformer_mocap_output_length_2 / mocap_fps * audio_sample_rate)

transformer_audio_waveform_input_length
transformer_audio_waveform_output_length

# the input length for the transformer is the same for mocap and audio
# since the transformer_mocap_input_length equals the number of latents for both modalities
pos_encoding_max_length = transformer_mocap_input_length


"""
Dataset Settings
"""

mocap_offset = 4
batch_size = 32 # 128
test_percentage = 0.1

"""
Training Settings
"""

learning_rate = 1e-4
non_teacher_forcing_step_count = 10
model_save_interval = 50
load_weights = False
save_weights = True
transformer_load_weights_path = "results_Diane_Take2_kld1.0_60hz/weights/transformer_weights_epoch_200"
epochs = 200

"""
Mocap Visualisation Settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

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
    
# get mocap info

mocap_skeleton = mocap_all_data[0]["skeleton"]

offsets = mocap_skeleton["offsets"].astype(np.float32)
parents = mocap_skeleton["parents"]
children = mocap_skeleton["children"]

mocap_motion = mocap_all_data[0]["motion"]["rot_local"]

mocap_joint_count = mocap_motion.shape[1]
mocap_joint_dim = mocap_motion.shape[2]
mocap_pose_dim = mocap_joint_count * mocap_joint_dim

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
for pose_count in range(8, 256):
    
    sample_count = pose_count * audio_samples_per_mocap_frame
    
    dummy_wave = torch.zeros((1, sample_count)).to(device)
    dummy_mels = vocos.feature_extractor(dummy_wave)
    dummy_wave2 = vocos.decode(dummy_mels)
    
    if dummy_wave.shape[-1] == dummy_wave2.shape[-1] and dummy_mels.shape[-1] % audio_mel_count_vae == 0:
        print("pc ", pose_count)
        print("dummy_mels s ", dummy_mels.shape)
"""

"""
Load Motion Autoencoder
"""

"""
Load Motion Autoencoder - Encoder
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
motion_encoder.eval()

motion_encoder.load_state_dict(torch.load(motion_vae_encoder_weights_file, map_location=device))

# test motion encoder

motion_encoder_input = torch.zeros((batch_size, motion_vae_mocap_length, mocap_pose_dim), dtype=torch.float32).to(device)
motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(motion_encoder_input)
motion_encoder_output = motion_encoder.reparameterize(motion_encoder_output_mu, torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8)

print("motion_encoder_input s ", motion_encoder_input.shape)
print("motion_encoder_output s ", motion_encoder_output.shape)

"""
Load Motion Autoencoder - Decoder
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
motion_decoder.eval()

motion_decoder.load_state_dict(torch.load(motion_vae_decoder_weights_file, map_location=device))

# test motion decoder

motion_decoder_input = motion_encoder_output
motion_decoder_output = motion_decoder(motion_decoder_input)

print("motion_decoder_input s ", motion_decoder_input.shape)
print("motion_decoder_output s ", motion_decoder_output.shape)

"""
Load Audio Autoencoder
"""

"""
Load Audio Autoencoder - Encoder
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
audio_encoder.eval()

audio_encoder.load_state_dict(torch.load(audio_vae_encoder_weights_file, map_location=device))

# test audio encoder

test_audio_waveform = torch.zeros((batch_size, audio_vocos_waveform_length)).to(device)
test_audio_mels = vocos.feature_extractor(test_audio_waveform)
audio_encoder_input = test_audio_mels.unsqueeze(1)
audio_encoder_input = audio_encoder_input[:, :, :, -audio_vae_mel_count:]

audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
audio_encoder_output = audio_encoder.reparameterize(audio_encoder_output_mu, torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8)

print("audio_encoder_input s ", audio_encoder_input.shape)
print("audio_encoder_output s ", audio_encoder_output.shape)

"""
Load Audio VAE Decoder - Decoder
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
        
        #print("stride ", stride)
                
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
audio_decoder.eval()

audio_decoder.load_state_dict(torch.load(audio_vae_decoder_weights_file, map_location=device))

# test audio decoder
audio_decoder_input = audio_encoder_output
audio_decoder_output = audio_decoder(audio_decoder_input)

print("audio_decoder_input s ", audio_decoder_input.shape)
print("audio_decoder_output s ", audio_decoder_output.shape)
    
"""
Compute Motion Latents Statistics
"""

if load_motion_latents_stats == True:
    
    motion_latents_mean = torch.load(motion_latents_mean_file)
    motion_latents_std = torch.load(motion_latents_std_file)
    
    motion_latents_mean = motion_latents_mean.to(device)
    motion_latents_std = motion_latents_std.to(device)

else:
    with torch.no_grad():
        
        motion_latents_all = []
        
        for mocap_data in mocap_all_data:
            
            mocap_motion = mocap_data["motion"]["rot_local"]
            mocap_motion = torch.from_numpy(mocap_motion).to(torch.float32)
            mocap_motion = mocap_motion.reshape((-1, mocap_pose_dim))
            
            mocap_frame_count = mocap_motion.shape[0]

            for mfI in range(mocap_frame_start_offset, mocap_frame_count - motion_vae_mocap_length * batch_size, batch_size):
               
                #print("mfI ", mfI, " out of ", (mocap_frame_count - motion_vae_mocap_length)) 
               
                # get mocap sequence
                mocap_excerpt_start = mfI
                mocap_excerpt_end = mfI + motion_vae_mocap_length * batch_size
                mocap_excerpt = mocap_motion[mocap_excerpt_start:mocap_excerpt_end, :].to(device)
                
                #print("mocap_excerpt s ", mocap_excerpt.shape)
                
                # normalise mocap excerpt
                mocap_excerpt_norm = (mocap_excerpt - mocap_mean) / (mocap_std + 1e-8) 
                
                #print("mocap_excerpt_norm s ", mocap_excerpt_norm.shape)
                
                mocap_excerpt_norm = mocap_excerpt_norm.reshape((batch_size, motion_vae_mocap_length, mocap_pose_dim))
                
                #print("mocap_excerpt_norm 2 s ", mocap_excerpt_norm.shape)
                
                motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(mocap_excerpt_norm)
                motion_latent = motion_encoder.reparameterize(motion_encoder_output_mu, torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8)

                #print("motion_latent s ", motion_latent.shape)

                motion_latent = motion_latent.detach().cpu()
                motion_latents_all.append(motion_latent)
                
        motion_latents_all = torch.cat(motion_latents_all, dim=0)
        
        #print("motion_latents_all s ", motion_latents_all.shape)
        
        motion_latents_mean = torch.mean(motion_latents_all, dim=0, keepdim=True)
        motion_latents_std = torch.std(motion_latents_all, dim=0, keepdim=True)
        
        #print("motion_latents_mean s ", motion_latents_mean.shape)
        #print("motion_latents_std s ", motion_latents_std.shape)
        
        torch.save(motion_latents_mean.detach(), motion_latents_mean_file)
        torch.save(motion_latents_std.detach(), motion_latents_std_file)
        
        motion_latents_mean = motion_latents_mean.to(device)
        motion_latents_std = motion_latents_std.to(device)

print("motion_latents_mean s ", motion_latents_mean.shape)
print("motion_latents_std s ", motion_latents_std.shape)

"""
Compute Audio Latents Statistics
"""

if load_audio_latents_stats == True:
    audio_latents_mean = torch.load(audio_latents_mean_file)
    audio_latents_std = torch.load(audio_latents_std_file)
    
    audio_latents_mean = audio_latents_mean.to(device)
    audio_latents_std = audio_latents_std.to(device)
    
else:
    
    with torch.no_grad():
        
        audio_latents_all = []
        
        for audio_waveform_data in audio_all_data: 
            audio_mels = vocos.feature_extractor(audio_waveform_data.to(device))
            
            #print("audio_mels s ", audio_mels.shape)
            
            audio_mel_count = audio_mels.shape[-1]
            
            for amI in range(0, audio_mel_count - audio_vae_mel_count * batch_size, batch_size):
                
                audio_mels_excerpt = audio_mels[:, :, amI:amI + audio_vae_mel_count * batch_size]
                
                #print("amI ", amI, " audio_mels_excerpt s ", audio_mels_excerpt.shape)
                
                audio_encoder_in = audio_mels_excerpt.reshape((1, audio_mel_filter_count, batch_size, audio_vae_mel_count))
            
                #print("amI ", amI, " audio_encoder_in s ", audio_encoder_in.shape)
                
                audio_encoder_in = audio_encoder_in.permute((2, 0, 1, 3))
                
                #print("amI ", amI, " audio_encoder_in 2 s ", audio_encoder_in.shape)
                
                audio_encoder_out_mu, audio_encoder_out_std = audio_encoder(audio_encoder_in)
                audio_encoder_out_std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
                audio_encoder_out = audio_encoder.reparameterize(audio_encoder_out_mu, audio_encoder_out_std)
            
                #print("amI ", amI, " audio_encoder_out s ", audio_encoder_out.shape)
                
                audio_latents_all.append(audio_encoder_out.detach().cpu())
                
        audio_latents_all = torch.cat(audio_latents_all, dim=0)
        
        #print("audio_latents_all s ", audio_latents_all.shape)
        
        audio_latents_mean = torch.mean(audio_latents_all, dim=0, keepdim=True)
        audio_latents_std = torch.std(audio_latents_all, dim=0, keepdim=True)
        
        #print("audio_latents_mean s ", audio_latents_mean.shape)
        #print("audio_latents_std s ", audio_latents_std.shape)
        
        torch.save(audio_latents_mean.detach(), audio_latents_mean_file)
        torch.save(audio_latents_std.detach(), audio_latents_std_file)
        
        audio_latents_mean = audio_latents_mean.to(device)
        audio_latents_std = audio_latents_std.to(device)

print("audio_latents_mean s ", audio_latents_mean.shape)
print("audio_latents_std s ", audio_latents_std.shape)

"""
Create Dataset
"""

motion_mocap_dataset = []
audio_waveform_dataset = []

transformer_motion_mocap_full_seq_length = transformer_mocap_input_length + transformer_mocap_output_length_2
transformer_audio_waveform_full_seq_length = transformer_motion_mocap_full_seq_length  * audio_waveform_length_per_mocap_frame + audio_waveform_start_offset

for sI in range(len(mocap_all_data)):
    
    with torch.no_grad():
        
        motion_mocap = mocap_all_data[sI]["motion"]["rot_local"].reshape(-1, mocap_pose_dim)
        audio_waveform = audio_all_data[sI][0].reshape(1, -1)
    
        #print(sI)
        #print("motion_mocap s ", motion_mocap.shape)
        #print("audio_waveform s ", audio_waveform.shape)
        
        mocap_frame_count = motion_mocap.shape[0]
        
        for mfI in range(mocap_frame_start_offset, mocap_frame_count - transformer_motion_mocap_full_seq_length, mocap_offset):
            
            #print("mfI ", mfI, " out of ", (mocap_frame_count - transformer_motion_mocap_full_seq_length))
            
            # get motion mocap excerpt
            mocap_excerpt_start = mfI
            mocap_excerpt_end = mfI + transformer_motion_mocap_full_seq_length
            motion_mocap_excerpt = motion_mocap[mocap_excerpt_start:mocap_excerpt_end, ...]
            motion_mocap_excerpt = torch.from_numpy(motion_mocap_excerpt).to(torch.float32)
            
            #print("motion_mocap_excerpt s ", motion_mocap_excerpt.shape)
            
            # get audio waveform excerpt
            asI = mfI * audio_waveform_length_per_mocap_frame - audio_waveform_start_offset
            audio_waveform_start = asI
            audio_waveform_end = audio_waveform_start + transformer_audio_waveform_full_seq_length
            audio_waveform_excerpt = audio_waveform[0, audio_waveform_start:audio_waveform_end]
            
            #print("audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
            
            motion_mocap_dataset.append(motion_mocap_excerpt)
            audio_waveform_dataset.append(audio_waveform_excerpt)

motion_mocap_dataset = torch.stack(motion_mocap_dataset, dim=0)
audio_waveform_dataset = torch.stack(audio_waveform_dataset, dim=0)

print("motion_mocap_dataset s ", motion_mocap_dataset.shape)
print("audio_waveform_dataset s ", audio_waveform_dataset.shape)


class MotionAudioDataset(Dataset):
    def __init__(self, motion_mocap, audio_waveform):
        self.motion_mocap = motion_mocap
        self.audio_waveform = audio_waveform

    def __len__(self):
        return self.motion_mocap.shape[0]
    
    def __getitem__(self, idx):
        return self.motion_mocap[idx, ...], self.audio_waveform[idx, ...]

full_dataset = MotionAudioDataset(motion_mocap_dataset, audio_waveform_dataset)

item_motion_mocap, item_audio_waveform = full_dataset[0]

print("item_motion_mocap s ", item_motion_mocap.shape)
print("item_audio_waveform s ", item_audio_waveform.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

batch_motion_mocap, batch_audio_waveoform = next(iter(train_loader))

print("batch_motion_mocap s ", batch_motion_mocap.shape)
print("batch_audio_waveoform s ", batch_audio_waveoform.shape)

"""
Create Models - PositionalEncoding
"""

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


"""
Create Models - Transformer
"""

class Transformer(nn.Module):

    # Constructor
    def __init__(
        self,
        mocap_dim,
        audio_dim,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        pos_encoding_max_length
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # LAYERS
        self.mocap2embed = nn.Linear(mocap_dim, embed_dim) # map mocap data to embedding
        self.audio2embed = nn.Linear(audio_dim, embed_dim) # map audio data to embedding

        self.positional_encoder = PositionalEncoding(
            dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_decoder_layers)
        
        self.embed2audio = nn.Linear(embed_dim, audio_dim) # map embedding to audio data
        
    def get_src_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.ones(size, size)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask
       
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
        
       
    def forward(self, mocap_data, audio_data):
        
        #print("forward")
        
        #print("data s ", data.shape)

        src_mask = self.get_src_mask(mocap_data.shape[1]).to(mocap_data.device)
        tgt_mask = self.get_tgt_mask(audio_data.shape[1]).to(audio_data.device)
        
        mocap_embedded = self.mocap2embed(mocap_data) * math.sqrt(self.embed_dim)
        mocap_embedded = self.positional_encoder(mocap_embedded)
        
        audio_embedded = self.audio2embed(audio_data) * math.sqrt(self.embed_dim)
        audio_embedded = self.positional_encoder(audio_embedded)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        encoder_out = self.encoder(mocap_embedded, mask=src_mask)
        decoder_out = self.decoder(audio_embedded, encoder_out, tgt_mask =tgt_mask)
        
        out = self.embed2audio(decoder_out)
        
        return out


transformer = Transformer(mocap_dim=motion_vae_latent_dim, 
                          audio_dim=audio_vae_latent_dim,
                          embed_dim=transformer_embed_dim, 
                          num_heads=transformer_head_count, 
                          num_encoder_layers=transformer_layer_count, 
                          num_decoder_layers=transformer_layer_count, 
                          dropout_p=transformer_dropout,
                          pos_encoding_max_length=pos_encoding_max_length).to(device)

print(transformer)

if load_weights and transformer_load_weights_path:
    transformer.load_state_dict(torch.load(transformer_load_weights_path, map_location=device))

# test transformer model

batch_motion_mocap, batch_audio_waveform = next(iter(train_loader))

batch_motion_mocap = batch_motion_mocap.to(device)
batch_audio_waveform = batch_audio_waveform.to(device)

print("batch_motion_mocap s ", batch_motion_mocap.shape)
print("batch_audio_waveform s ", batch_audio_waveform.shape)

transformer_motion_latents = []
transformer_audio_latents = []

for mfI in range(0, transformer_mocap_input_length + transformer_mocap_output_length, 1):
    
    print("mfI ", mfI)
    
    mocap_motion_excerpt = batch_motion_mocap[:, mfI:mfI + motion_vae_mocap_length, ...]

    print("mocap_motion_excerpt s ", mocap_motion_excerpt.shape)
    
    mocap_motion_excerpt_norm = (mocap_motion_excerpt - mocap_mean.unsqueeze(0)) / (mocap_std.unsqueeze(0) + 1e-8)
    motion_encoder_input = mocap_motion_excerpt_norm
    
    print("motion_encoder_input s ", motion_encoder_input.shape)
    
    motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(motion_encoder_input)
    mu = motion_encoder_output_mu
    std = torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8
    motion_latents = motion_encoder.reparameterize(mu, std)
    
    print("motion_latents s ", motion_latents.shape)
    
    motion_latents_norm = (motion_latents - motion_latents_mean) / (motion_latents_std + 1e-8)
    
    print("motion_latents_norm s ", motion_latents_norm.shape)
    
    transformer_motion_latents.append(motion_latents_norm)
    
    asI = mfI * audio_waveform_length_per_mocap_frame
    
    print("asI ", asI)
    
    audio_waveform_excerpt = batch_audio_waveform[:, asI:asI+audio_vocos_waveform_length]
    
    print("audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
    
    audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
    
    print("audio_mels_excerpt s ", audio_mels_excerpt.shape)
    
    audio_mels_excerpt = audio_mels_excerpt[:, :, -audio_vae_mel_count:]
    
    print("audio_mels_excerpt 2 s ", audio_mels_excerpt.shape)
    
    audio_mels_excerpt_norm = (audio_mels_excerpt - audio_mean) / (audio_std + 1e-8)
    audio_encoder_input = audio_mels_excerpt_norm.unsqueeze(1)
    
    print("audio_encoder_input s ", audio_encoder_input.shape)
    
    audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
    mu = audio_encoder_output_mu
    std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
    audio_latents = audio_encoder.reparameterize(mu, std)
    
    print("audio_latents s ", audio_latents.shape)
    
    audio_latents_norm = (audio_latents - audio_latents_mean) / (audio_latents_std + 1e-8)
    
    print("audio_latents_norm s ", audio_latents_norm.shape)
    
    transformer_audio_latents.append(audio_latents_norm)
    
transformer_motion_latents = torch.stack(transformer_motion_latents, dim=1)
transformer_audio_latents = torch.stack(transformer_audio_latents, dim=1)

print("transformer_motion_latents s ", transformer_motion_latents.shape)
print("transformer_audio_latents s ", transformer_audio_latents.shape)

transformer_input_motion_latents = transformer_motion_latents[:, :transformer_mocap_input_length, :]
transformer_input_audio_latents = transformer_audio_latents[:, :transformer_mocap_input_length, :]
transformer_target_audio_latents = transformer_audio_latents[:, 1:1 + transformer_mocap_input_length, :]

transformer_output_audio_latents = transformer(transformer_input_motion_latents, transformer_input_audio_latents)

print("transformer_input_motion_latents s ", transformer_input_motion_latents.shape)
print("transformer_input_audio_latents s ", transformer_input_audio_latents.shape)
print("transformer_target_audio_latents s ", transformer_target_audio_latents.shape)
print("transformer_output_audio_latents s ", transformer_output_audio_latents.shape)


"""
Training
"""

optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) # reduce the learning every 20 epochs by a factor of 10

n1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

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

def audio_latents_loss(y_latents, yhat_latents):
    
    _all = mse_loss(yhat_latents, y_latents)

    return _all

def audio_mel_loss(y_mels, yhat_mels):
    
    _aml = mse_loss(yhat_mels, y_mels)

    return _aml

def audio_perc_loss(y_wave, yhat_wave):
    
    apl = perc_loss(yhat_wave, y_wave)
    
    return apl

"""
Training and Test Steps
"""    

"""
Motion2Audio Transformer Training and Test Step
"""

motion_latents_std.shape


def transformer_train_step(motion_mocap, audio_waveform):
    
    print("motion_mocap s ", motion_mocap.shape)
    print("audio_waveform s ", audio_waveform.shape)
    
    transformer_audio_output = None
    _avg_loss = 0.0
    
    # outer loop
    for mfI1 in range(0, transformer_mocap_output_length, 1):
        
        # inner loop
        motion_latents_norm = []
        audio_latents_norm = []
        
        for mfI2 in range(0, transformer_mocap_input_length + 1, 1):
                        
            # motion mocap excerpt
            mfI = mfI1 + mfI2
            print("mfI1 ", mfI1, " mfI2 ", mfI2, " mfI ", mfI)
            motion_mocap_excerpt = motion_mocap[:, mfI:mfI + motion_vae_mocap_length, ... ]
            print("motion_mocap_excerpt s ", motion_mocap_excerpt.shape)
            
            # audio waveform excerpt
            asI = mfI * audio_waveform_length_per_mocap_frame
            print("asI ", asI)
            audio_waveform_excerpt = audio_waveform[:, asI:asI+audio_vocos_waveform_length]
            print("audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
            
            # convert audio waveform excerpt to audio mels
            audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
            print("audio_mels_excerpt s ", audio_mels_excerpt.shape)
            
            # get last audio_vae_mel_count mels
            audio_mels_excerpt = audio_mels_excerpt[:, :, -audio_vae_mel_count:]
            print("audio_mels_excerpt 2 s ", audio_mels_excerpt.shape)
            
            # normalise motion mocap excerpt
            motion_mocap_norm_excerpt = (motion_mocap_excerpt - mocap_mean.unsqueeze(0)) / (mocap_std + 1e-8)
            print("motion_mocap_norm_excerpt s ", motion_mocap_excerpt.shape)
            
            # normalise audio mels excerpt
            audio_mels_norm_excerpt = (audio_mels_excerpt - audio_mean) / (audio_std + 1e-8)
            print("audio_mels_norm_excerpt s ", audio_mels_norm_excerpt.shape)
            
            # calculate motion latent
            motion_encoder_input = motion_mocap_norm_excerpt
            motion_encoder_output_mu, motion_encoder_output_std = motion_encoder(motion_encoder_input)
            mu = motion_encoder_output_mu
            std = torch.nn.functional.softplus(motion_encoder_output_std) + 1e-8
            motion_latent = motion_encoder.reparameterize(mu, std)
            print("motion_latent s ", motion_latent.shape)
            
            # calculate audio latents
            audio_encoder_input = audio_mels_norm_excerpt.unsqueeze(1)
            audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
            mu = audio_encoder_output_mu
            std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
            audio_latent = audio_encoder.reparameterize(mu, std)
            print("audio_latent s ", audio_latent.shape)
            
            # normalise motion latent
            motion_latent_norm = (motion_latent - motion_latents_mean) / (motion_latents_std + 1e-8)
            print("motion_latent_norm s ", motion_latent_norm.shape)
            
            # normalise audio latent
            audio_latent_norm = (audio_latent - audio_latents_mean) / (audio_latents_std + 1e-8)
            print("audio_latent_norm s ", audio_latent_norm.shape)
            
            # append motion latent
            motion_latents_norm.append(motion_latent_norm)
            
            # append audio latent
            audio_latents_norm.append(audio_latent_norm)
            
        # stack motion latents norm 
        motion_latents_norm = torch.stack(motion_latents_norm, dim=1)
        print("motion_latents_norm s ", motion_latents_norm.shape)
        
        # stack audio latents norm 
        audio_latents_norm = torch.stack(audio_latents_norm, dim=1)
        print("audio_latents_norm s ", audio_latents_norm.shape)
            
        # get transformer motion input
        transformer_motion_input = motion_latents_norm[:, :transformer_mocap_input_length, :]
        print("transformer_motion_input s ", transformer_motion_input.shape)
        
        # get transformer audio input
        if mfI1 == 0: # teacher forcing
            transformer_audio_input = audio_latents_norm[:, :transformer_mocap_input_length, :]
        else:
            transformer_audio_input = transformer_audio_output.detach()
        print("transformer_audio_input s ", transformer_audio_input.shape)
        
        # get transformer audio target
        transformer_audio_target = audio_latents_norm[:, 1:transformer_mocap_input_length+1, :]
        print("transformer_audio_target s ", transformer_audio_target.shape)
        
        # calculate transformer audio output
        transformer_audio_output = transformer(transformer_motion_input, transformer_audio_input)
        print("transformer_audio_output s ", transformer_audio_output.shape)
        
        # compute audio latent reconstruction loss
        _audio_latents_loss = audio_latents_loss(transformer_audio_target.reshape(-1, audio_vae_latent_dim), transformer_audio_output.reshape(-1, audio_vae_latent_dim))
        
        # compute audio mel reconstruction loss
        transformer_audio_mels_norm_target = audio_decoder(transformer_audio_target.reshape(-1, audio_vae_latent_dim))
        transformer_audio_mels_norm_output = audio_decoder(transformer_audio_output.reshape(-1, audio_vae_latent_dim))  
        
        print("transformer_audio_mels_norm_target s ", transformer_audio_mels_norm_target.shape)
        print("transformer_audio_mels_norm_output s ", transformer_audio_mels_norm_output.shape)
        
        # calculate losses and perform backprop
        _loss = _audio_latents_loss

        # Backpropagation
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()

        _avg_loss += _loss.detach().cpu().item()
        
        return _avg_loss

            
# test transformer train step

transformer_motion_mocap, transformer_audio_waveform = next(iter(train_loader))
transformer_motion_mocap = transformer_motion_mocap.to(device)
transformer_audio_waveform = transformer_audio_waveform.to(device)

_ = transformer_train_step(transformer_motion_mocap, transformer_audio_waveform)





x_mocap_batch = x_mocap_batch.to(device)
x_audio_batch = x_audio_batch.to(device)
y_audio_batch = y_audio_batch.to(device)
_loss = train_step(x_mocap_batch, x_audio_batch, y_audio_batch)


def audio_loss(y_latents_norm, yhat_latents_norm):
    
    print("loss begin")
    
    print("y_latents_norm s ", y_latents_norm.shape, " yhat_latents_norm s ", yhat_latents_norm.shape)
    
    batch_size = y_latents_norm.shape[0]
    
    print("batch_size s ", batch_size)
    
    y_latents_norm = y_latents_norm.reshape((-1, audio_vae_latent_dim))
    yhat_latents_norm = yhat_latents_norm.reshape((-1, audio_vae_latent_dim))
    
    print("y_latents_norm 2 s ", y_latents_norm.shape, " yhat_latents_norm 2 s ", yhat_latents_norm.shape)
    
    # calculate audio latent loss
    #start_time = time.time()
    
    _audio_latents_loss = audio_latents_loss(y_latents_norm, yhat_latents_norm)
    
    #end_time = time.time()
    #elapsed_seconds = end_time - start_time
    #print(f"calculate audio latent loss: Elapsed time: {elapsed_seconds:.4f} seconds")
    
    #print("1")
    
    # convert latents norm to latents
    #start_time = time.time()
    
    y_latents = y_latents_norm * audio_latents_std + audio_latents_mean
    yhat_latents = yhat_latents_norm * audio_latents_std + audio_latents_mean
    
    print("y_latents s ", y_latents.shape, " yhat_latents s ", yhat_latents.shape)
    
    # decode latents into normalised mels
    y_mels_norm = audio_decoder(y_latents)
    yhat_mels_norm = audio_decoder(yhat_latents)
    
    print("y_mels_norm s ", y_mels_norm.shape, " yhat_mels_norm s ", yhat_mels_norm.shape)
    
    # calculate audio mel loss
    y_mels_norm_2 = y_mels_norm.squeeze(1).permute(0, 2, 1)
    _audio_mel_loss = audio_mel_loss()
    
    
    # denormalize mels
    y_mels_norm = y_mels_norm.squeeze(1)
    yhat_mels_norm = yhat_mels_norm.squeeze(1)
    
    print("y_mels_norm 2 s ", y_mels_norm.shape, " yhat_mels_norm 2 s ", yhat_mels_norm.shape)
    
    y_mels = y_mels_norm * audio_std + audio_mean
    yhat_mels = yhat_mels_norm * audio_std + audio_mean
    
    print("y_mels s ", y_mels.shape, " yhat_mels s ", yhat_mels.shape)
    
    
    #print("2")
    """
    print("y_mels_regrouped s ", y_mels_regrouped.shape, " yhat_mels_regrouped s ", yhat_mels_regrouped.shape)
    

    # regroup mels
    y_mels = y_mels_regrouped.reshape(batch_size, -1, 1, audio_mel_filter_count, audio_vae_mel_count)
    yhat_mels = yhat_mels_regrouped.reshape(batch_size, -1, 1, audio_mel_filter_count, audio_vae_mel_count)
    
    print("y_mels s ", y_mels.shape, " yhat_mels s ", yhat_mels.shape)

    y_mels = y_mels.permute(0, 2, 3, 1, 4) 
    yhat_mels = yhat_mels.permute(0, 2, 3, 1, 4) 
    
    print("y_mels 2 s ", y_mels.shape, " yhat_mels 2 s ", yhat_mels.shape)
    
    y_mels = y_mels.reshape(batch_size, audio_mel_filter_count, -1) 
    yhat_mels = yhat_mels.reshape(batch_size, audio_mel_filter_count, -1) 
    
    print("y_mels 3 s ", y_mels.shape, " yhat_mels 3 s ", yhat_mels.shape)
    
    # calculate audio mel loss
    _audio_mel_loss = audio_mel_loss(y_mels, yhat_mels)
    
    #end_time = time.time()
    #elapsed_seconds = end_time - start_time
    #print(f"calculate audio mel loss: Elapsed time: {elapsed_seconds:.4f} seconds")
    
    #print("3")
    
    # convert audio mels to audio waveforms
    #start_time = time.time()
    
    y_waveform = vocos.decode(y_mels)
    yhat_waveform = vocos.decode(yhat_mels)
    
    #print("4")
    
    print("y_waveform s ", y_waveform.shape, " yhat_waveform s ", yhat_waveform.shape)
    
    y_waveform = y_waveform.unsqueeze(1)
    yhat_waveform = yhat_waveform.unsqueeze(1)
    
    print("y_waveform 2 s ", y_waveform.shape, " yhat_waveform 2 s ", yhat_waveform.shape)
    
    # calculate audio perceptual loss
    _audio_perc_loss = audio_perc_loss(y_waveform, yhat_waveform)
    
    #end_time = time.time()
    #elapsed_seconds = end_time - start_time
    #print(f"calculate audio perceptual loss: Elapsed time: {elapsed_seconds:.4f} seconds")
    
    #print("5")
    
    _total_loss = 0.0
    _total_loss += _audio_latents_loss * 0.33
    _total_loss += _audio_mel_loss * 0.33
    _total_loss += _audio_perc_loss * 0.33
    
    #print("loss end")
    
    return _total_loss
    """


"""
# test audio loss function
_ = audio_loss(transformer_target_audio_latents, transformer_output_audio_latents)
"""

def train_step(x_mocap, x_audio, y_audio):

    #print("x_mocap s ", x_mocap.shape)
    #print("x_audio s ", x_audio.shape)
    #print("y_audio s ", y_audio.shape)
    
    # first step (teacher forcing)
    
    transformer_mocap_input = x_mocap[:, :mocap_input_seq_length, ...]
    transformer_audio_input = x_audio
    transformer_audio_target = y_audio[:, 0, ...]
    
    #print("transformer_mocap_input s ", transformer_mocap_input.shape)
    #print("transformer_audio_input s ", transformer_audio_input.shape)
    #print("transformer_audio_target s ", transformer_audio_target.shape)
    
    transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

    # todo: also compute mel loss and waveform perceptual loss
    step_loss = loss(transformer_audio_target, transformer_audio_output) 
    _loss = step_loss.detach().cpu()
    
    # Backpropagation
    optimizer.zero_grad()
    step_loss.backward()
    optimizer.step()
    
    # next steps(non-teacher forcing)
    for mfI in range(1, mocap_output_seq_length):
        
        #print("mfI ", mfI)
        
        transformer_mocap_input = x_mocap[:, mfI:mocap_input_seq_length + mfI, ...].to(device)
        transformer_audio_input = transformer_audio_output.detach().clone()
        transformer_audio_target = y_audio[:, mfI, ...]
        
        #print("transformer_mocap_input s ", transformer_mocap_input.shape)
        #print("transformer_audio_input s ", transformer_audio_input.shape)
        #print("transformer_audio_target s ", transformer_audio_target.shape)
        
        transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

        step_loss = loss(transformer_audio_target, transformer_audio_output) 
        _loss += step_loss.detach().cpu()

        # Backpropagation
        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()
        
    _loss /= (mocap_output_seq_length + 1)

    return _loss

"""
x_mocap_batch, x_audio_batch, y_audio_batch = next(iter(train_loader))
x_mocap_batch = x_mocap_batch.to(device)
x_audio_batch = x_audio_batch.to(device)
y_audio_batch = y_audio_batch.to(device)
_loss = train_step(x_mocap_batch, x_audio_batch, y_audio_batch)
"""

@torch.no_grad()
def test_step(x_mocap, x_audio, y_audio):
    
    transformer.eval()

    #print("x_mocap s ", x_mocap.shape)
    #print("x_audio s ", x_audio.shape)
    #print("y_audio s ", y_audio.shape)
    
    # first step (teacher forcing)
    
    transformer_mocap_input = x_mocap[:, :mocap_input_seq_length, ...]
    transformer_audio_input = x_audio
    transformer_audio_target = y_audio[:, 0, ...]
    
    #print("transformer_mocap_input s ", transformer_mocap_input.shape)
    #print("transformer_audio_input s ", transformer_audio_input.shape)
    #print("transformer_audio_target s ", transformer_audio_target.shape)
    
    transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

    # todo: also compute mel loss and waveform perceptual loss
    step_loss = loss(transformer_audio_target, transformer_audio_output) 
    _loss = step_loss.detach().cpu()

    # next steps(non-teacher forcing)
    for mfI in range(1, mocap_output_seq_length):
        
        #print("mfI ", mfI)
        
        transformer_mocap_input = x_mocap[:, mfI:mocap_input_seq_length + mfI, ...].to(device)
        transformer_audio_input = transformer_audio_output.detach().clone()
        transformer_audio_target = y_audio[:, mfI, ...]
        
        #print("transformer_mocap_input s ", transformer_mocap_input.shape)
        #print("transformer_audio_input s ", transformer_audio_input.shape)
        #print("transformer_audio_target s ", transformer_audio_target.shape)
        
        transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

        step_loss = loss(transformer_audio_target, transformer_audio_output) 
        _loss += step_loss.detach().cpu()

    _loss /= (mocap_output_seq_length + 1)
    
    transformer.train()

    return _loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []

        for train_batch in train_dataloader:
            x_mocap = train_batch[0].to(device)
            x_audio = train_batch[1].to(device)
            y_audio = train_batch[2].to(device)
            
            _loss = train_step(x_mocap, x_audio, y_audio)
            
            _loss = _loss.detach().cpu().numpy()
            
            _train_loss_per_epoch.append(_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))

        _test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            x_mocap = test_batch[0].to(device)
            x_audio = test_batch[1].to(device)
            y_audio = test_batch[2].to(device)
            
            _loss = test_step(x_mocap, x_audio, y_audio)

            _loss = _loss.detach().cpu().numpy()
            
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(transformer.state_dict(), "results/weights/transformer_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(image_file_name)

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

# save model weights
torch.save(transformer.state_dict(), "results/weights/transformer_weights_epoch_{}".format(epochs))

# inference

audio_window_length = audio_samples_per_mocap_frame * 4
audio_window_env = torch.hann_window(audio_window_length)

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


def create_mocap_anim(mocap_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    pose_sequence = mocap_data[mocap_start_frame_index:mocap_start_frame_index + mocap_frame_count]

    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, joint_dim))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=1000 / mocap_fps, loop=0)
    

def create_orig_audio(waveform_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    audio_waveform_excerpt_start_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    audio_waveform_excerpt_end_index = audio_waveform_excerpt_start_index + mocap_frame_count * audio_samples_per_mocap_frame

    audio_waveform_excerpt = waveform_data[audio_waveform_excerpt_start_index:audio_waveform_excerpt_end_index]
    
    torchaudio.save(file_name, audio_waveform_excerpt.unsqueeze(0), audio_sample_rate)

@torch.no_grad()
def create_vocos_audio(waveform_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    audio_waveform_excerpt_start_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    audio_waveform_excerpt_sample_count = mocap_frame_count * audio_samples_per_mocap_frame
    audio_waveform_excerpt_end_index = audio_waveform_excerpt_start_index + audio_waveform_excerpt_sample_count

    orig_audio_waveform = waveform_data[audio_waveform_excerpt_start_index:audio_waveform_excerpt_end_index]

    gen_audio_waveform = torch.zeros((audio_waveform_excerpt_sample_count), dtype=torch.float32)
    
    #print("gen_audio_waveform s ", gen_audio_waveform.shape)
    
    for aSI in range(0, audio_waveform_excerpt_sample_count - audio_waveform_input_seq_length, audio_samples_per_mocap_frame):
        
        #print("aSI ", aSI)
        
        audio_waveform = orig_audio_waveform[aSI:aSI + audio_waveform_input_seq_length].to(device)

        #print("audio_waveform s ", audio_waveform.shape)

        audio_waveform = audio_waveform.reshape((1, audio_waveform_input_seq_length))
        audio_mels = vocos.feature_extractor(audio_waveform)
        
        #print("audio_mels_2 s ", audio_mels.shape)
        
        audio_waveform_2 = vocos.decode(audio_mels)
        
        #print("audio_waveform_2 s ", audio_waveform_2.shape)
        
        audio_waveform_2_window = audio_waveform_2.reshape(-1)[-audio_window_length:]
        audio_waveform_2_window = audio_waveform_2_window.detach().cpu()
        
        #print("audio_waveform_2_window s ", audio_waveform_2_window.shape)
        #print("audio_window_env s ", audio_window_env.shape)
        #print("gen_audio_waveform[aSI:aSI + audio_window_length] s ", gen_audio_waveform[aSI:aSI + audio_window_length].shape)
        
        gen_audio_waveform[aSI:aSI + audio_window_length] += audio_waveform_2_window * audio_window_env

        torchaudio.save(file_name, gen_audio_waveform.unsqueeze(0), audio_sample_rate)

@torch.no_grad()
def create_gen_audio(mocap_data, waveform_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    transformer.eval()
    
    # prepare mocap data
    mocap_end_frame_index = mocap_start_frame_index + mocap_frame_count
    mocap_data = mocap_data[mocap_start_frame_index:mocap_end_frame_index, ...].to(device)
    mocap_data_norm = (mocap_data - mocap_mean) / (mocap_std + 1e-8)
    mocap_data_norm = mocap_data_norm.unsqueeze(0)
    
    #print("mocap_data_norm s ", mocap_data_norm.shape)
    
    # prepare audio data

    audio_waveform_sample_count = mocap_frame_count * audio_samples_per_mocap_frame
    gen_audio_waveform = torch.zeros((audio_waveform_sample_count), dtype=torch.float32)

    audio_waveform_start_sample_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    audio_waveform_end_sample_index =  audio_waveform_start_sample_index + audio_waveform_input_seq_length
    audio_waveform_data = waveform_data[audio_waveform_start_sample_index:audio_waveform_end_sample_index]
    audio_waveform_data = audio_waveform_data.unsqueeze(0).to(device)
    
    #print("audio_waveform_data s ", audio_waveform_data.shape)

    audio_mels_data = vocos.feature_extractor(audio_waveform_data)
    
    #print("audio_mels_data s ", audio_mels_data.shape)
    
    audio_encoder_in = audio_mels_data.reshape((1, audio_mel_filter_count, -1, audio_mel_count_vae))
    
    #print("audio_encoder_in s ", audio_encoder_in.shape)
    
    audio_encoder_in = audio_encoder_in.permute((2, 0, 1, 3))
                
    #print("audio_encoder_in 2 s ", audio_encoder_in.shape)
    
    audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
    audio_encoder_out_std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
    audio_latents = encoder.reparameterize(audio_encoder_out_mu, audio_encoder_out_std)
    
    #print("audio_latents_data s ", audio_latents_data.shape)
    
    audio_latents_norm = (audio_latents - audio_latents_mean) / (audio_latents_std + 1e-8)
            
    #print("audio_latents_norm s ", audio_latents_norm.shape)
    
    audio_latents_norm = audio_latents_norm.unsqueeze(0)
    
    #print("audio_latents_norm 2 s ", audio_latents_norm.shape)
    
    # predict audio
    
    x_mocap = mocap_data_norm[:, :mocap_input_seq_length, ...]
    
    #print("x_mocap s ", x_mocap.shape)
    
    x_audio = audio_latents_norm
    
    #print("x_audio s ", x_audio.shape)
    
    yhat_audio = transformer(x_mocap, x_audio)
    
    #print("yhat_audio s ", yhat_audio.shape)

    yhat_audio_latents_norm = yhat_audio.detach().squeeze(0)
    
    #print("yhat_audio_latents_norm s ", yhat_audio_latents_norm.shape)
    
    yhat_audio_latents = yhat_audio_latents_norm * audio_latents_std + audio_latents_mean

    #print("yhat_audio_latents s ", yhat_audio_latents.shape)
    
    yhat_audio_mels = decoder(yhat_audio_latents)
    
    #print("yhat_audio_mels s ", yhat_audio_mels.shape)
    
    yhat_audio_mels = yhat_audio_mels.permute((1, 2, 0, 3))
    
    #print("yhat_audio_mels 2 s ", yhat_audio_mels.shape)
    
    yhat_audio_mels = yhat_audio_mels.reshape((1, audio_mel_filter_count, -1))
    
    #print("yhat_audio_mels 3 s ", yhat_audio_mels.shape)
    
    yhat_audio_waveform = vocos.decode(yhat_audio_mels)
    
    #print("yhat_audio_waveform s ", yhat_audio_waveform.shape)
    
    yhat_audio_window = yhat_audio_waveform.reshape(-1)[-audio_window_length:]
    yhat_audio_window = yhat_audio_window.detach().cpu()
    
    #print("yhat_audio_window s ", yhat_audio_window.shape)
    
    gen_audio_waveform[:audio_window_length] += yhat_audio_window * audio_window_env

    for mFI in range(1, mocap_frame_count - mocap_input_seq_length):  
        
        #print("mFI ", mFI)
    
        aSI = mFI * audio_samples_per_mocap_frame
        
        x_mocap = mocap_data_norm[:, mFI:mFI + mocap_input_seq_length, ...]
        x_audio = yhat_audio.detach()
        
        #print("x_mocap s ", x_mocap.shape)
        #print("x_audio s ", x_audio.shape)
        
        yhat_audio = transformer(x_mocap, x_audio)
        
        #print("yhat_audio s ", yhat_audio.shape)
        
        yhat_audio_latents_norm = yhat_audio.detach().squeeze(0)
        
        #print("yhat_audio_latents_norm s ", yhat_audio_latents_norm.shape)
        
        yhat_audio_latents = yhat_audio_latents_norm * audio_latents_std + audio_latents_mean

        #print("yhat_audio_latents s ", yhat_audio_latents.shape)
        
        yhat_audio_mels = decoder(yhat_audio_latents)
        
        #print("yhat_audio_mels s ", yhat_audio_mels.shape)
        
        yhat_audio_mels = yhat_audio_mels.permute((1, 2, 0, 3))
        
        #print("yhat_audio_mels 2 s ", yhat_audio_mels.shape)
        
        yhat_audio_mels = yhat_audio_mels.reshape((1, audio_mel_filter_count, -1))
        
        #print("yhat_audio_mels 3 s ", yhat_audio_mels.shape)
        
        yhat_audio_waveform = vocos.decode(yhat_audio_mels)
        
        #print("yhat_audio_waveform s ", yhat_audio_waveform.shape)
        
        yhat_audio_window = yhat_audio_waveform.reshape(-1)[-audio_window_length:]
        yhat_audio_window = yhat_audio_window.detach().cpu()
        
        #print("yhat_audio_window s ", yhat_audio_window.shape)
        
        gen_audio_waveform[aSI:aSI+ audio_window_length] += yhat_audio_window * audio_window_env

    torchaudio.save(file_name, gen_audio_waveform.unsqueeze(0), audio_sample_rate)
    
    transformer.train()
    
    
"""
generate audio with orig mocap data
"""
    
test_mocap_data = torch.from_numpy(mocap_all_data[0]["motion"]["rot_local"]).to(torch.float32)
test_mocap_data = test_mocap_data.reshape(-1, pose_dim)
test_audio_data = audio_all_data[0][0]

#print("test_mocap_data s ", test_mocap_data.shape)
#print("test_audio_data s ", test_audio_data.shape)

test_mocap_start_times = [100, 200, 300]
test_mocap_duration = 30
    
for test_mocap_start_time in test_mocap_start_times:
    create_mocap_anim(test_mocap_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/anims/orig_mocap_{}-{}.gif".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
    create_orig_audio(test_audio_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/audio/orig_audio_{}-{}.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
    create_vocos_audio(test_audio_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/audio/vocos_audio_{}-{}.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
    create_gen_audio(test_mocap_data, test_audio_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/audio/gen_audio_{}-{}_epoch_{}_orig.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration), epochs))


"""
generate audio with alternative mocap data
"""

test_mocap_data_path = "E:/data/mocap/Diane/Solos/ZHdK_10.10.2025/fbx_60hz/"

test_mocap_data_files = ["trial-002.fbx",
                         "trial-003.fbx",
                         "trial-004.fbx",
                         "trial-005.fbx",
                         "trial-006.fbx"]
test_mocap_valid_ranges = [[364, 22739],
                           [531, 22905],
                           [549, 22924],
                           [613, 22988],
                           [1171, 23545]]
test_mocap_start_times = [[100, 200, 300],
                          [100, 200, 300],
                          [100, 200, 300],
                          [100, 200, 300],
                          [100, 200, 300]]
test_mocap_duration = 30
test_audio_data = audio_all_data[0][0]



for fI in range(len(test_mocap_data_files)):
    
    test_mocap_data_file = test_mocap_data_files[fI]
    test_mocap_valid_range = test_mocap_valid_ranges[fI]
    test_mocap_start_times2 = test_mocap_start_times[fI]

    if test_mocap_data_file.endswith(".bvh") or test_mocap_data_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(test_mocap_data_path + test_mocap_data_file)
        test_mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif test_mocap_data_file.endswith(".fbx") or test_mocap_data_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(test_mocap_data_path + test_mocap_data_file)
        test_mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only  
        
    test_mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    test_mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
    # set x and z offset of root joint to zero
    test_mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    test_mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
    
    if test_mocap_data_file.endswith(".bvh") or test_mocap_data_file.endswith(".BVH"):
        test_mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(test_mocap_data["motion"]["rot_local_euler"], test_mocap_data["rot_sequence"])
    elif test_mocap_data_file.endswith(".fbx") or test_mocap_data_file.endswith(".FBX"):
        test_mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(test_mocap_data["motion"]["rot_local_euler"], test_mocap_data["rot_sequence"])
    
    test_mocap_data = torch.from_numpy(test_mocap_data["motion"]["rot_local"]).to(torch.float32)
    test_mocap_data = test_mocap_data.reshape(-1, pose_dim)
    

    for test_mocap_start_time in test_mocap_start_times2:
        create_mocap_anim(test_mocap_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/anims/test_mocap_{}_{}-{}.gif".format(test_mocap_data_file, test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
        create_gen_audio(test_mocap_data, test_audio_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/audio/test_gen_audio_{}_{}-{}_epoch_{}.wav".format(test_mocap_data_file, test_mocap_start_time, (test_mocap_start_time + test_mocap_duration), epochs))
    
