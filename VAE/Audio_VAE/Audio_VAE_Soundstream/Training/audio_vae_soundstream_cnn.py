"""
variational autoencoder for soundstream spectra
"""

"""
Imports
"""

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import torchaudio
import simpleaudio as sa
import numpy as np
import random
import glob
from matplotlib import pyplot as plt
import os, time
import json
import csv

import soundstream_utils as ssu

import auraloss


"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Audio Settings
"""

"""
audio_file_path = "E:/Data/audio/Diane/48khz/"
audio_files = ["4d69949b.wav"]
"""

audio_file_path = "E:/data/audio/Eleni/"
audio_files = ["4_5870821179501060412.wav"]

audio_orig_sample_rate = 48000 # numer of audio samples per sec
audio_channels = 1

audio_waveform_length_soundstream = 48000 # 80 specs worth of audio
audio_spec_count_soundstream = None # will be calculated
audio_spec_count_vae = 8


"""
VAE Model Settings
"""

latent_dim = 32
vae_conv_channel_counts = [ 16, 32, 64, 128 ]
vae_conv_kernel_size = (5, 3)
vae_dense_layer_sizes = [ 512 ]

save_weights = False
load_weights = True
encoder_weights_file = "results/weights/encoder_weights_epoch_150"
decoder_weights_file = "results/weights/decoder_weights_epoch_150"

"""
Training Settings
"""

data_count = 100000 # 100000
batch_size = 32


train_percentage = 0.9 # train / test split
test_percentage  = 0.1
ae_learning_rate = 1e-4
ae_rec_loss_scale = 5.0
ae_beta = 0.0 # will be calculated
ae_beta_cycle_duration = 100
ae_beta_min_const_duration = 20
ae_beta_max_const_duration = 20
ae_min_beta = 0.0
ae_max_beta = 0.1


epochs = 400
model_save_interval = 50
save_history = True

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

set_all_seeds(42)

"""
Load Audio
"""

audio_all_data = []

for audio_file in audio_files:   
    
    audio_waveform, _ = torchaudio.load(audio_file_path + audio_file)
  
    print("audio_file ", audio_file ," audio_waveform s ", audio_waveform.shape)
    
    audio_all_data.append(audio_waveform)

"""
Create Soundstream Model
"""

# first create sound stream backend without audio stats
soundstream_nostat = ssu.SoundStreamSpecBackend(
        device=device,
        audio_sample_rate=audio_orig_sample_rate,
        target_sr=ssu.soundstream_audio_sample_rate,
        min_len_16k=ssu.min_soundstream_samples,
    )

# calculate soundstream normalization statistics
q_mean, q_std = ssu.compute_soundstream_norm_stats(soundstream_nostat, 
                                               audio_all_data, 
                                               audio_orig_sample_rate, 
                                               ssu.hop_length, 
                                               ssu.max_windows_per_clip)

# create sound stream backend with audio stats
soundstream = ssu.SoundStreamSpecBackend(
        device=device,
        audio_sample_rate=audio_orig_sample_rate,      # uses your global config
        target_sr=ssu.soundstream_audio_sample_rate,
        min_len_16k=ssu.min_soundstream_samples,
        q_mean=q_mean ,
        q_std=q_std,
    )

dummy_wave = torch.zeros((1, 48000 // 2)).to(device)
dummy_wave_rs = torchaudio.functional.resample(dummy_wave, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
dummy_spec = soundstream.wav_to_spec(dummy_wave_rs.to(device))
dummy_wave_rs2 = soundstream.spec_to_wav(dummy_spec)
dummy_wave2 = torchaudio.functional.resample(dummy_wave_rs2,  ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)


print("dummy_wave s ", dummy_wave.shape)
print("dummy_wave_rs s ", dummy_wave_rs.shape)
print("dummy_spec s ", dummy_spec.shape)
print("dummy_wave_rs2 s ", dummy_wave_rs2.shape)
print("dummy_wave2 s ", dummy_wave2.shape)

"""
# debug
audio_samples_per_mocap_frame = 800 # mocap 60 fps
for pose_count in range(1, 256):
    
    sample_count = pose_count * audio_samples_per_mocap_frame
    
    dummy_wave = torch.zeros((1, sample_count)).to(device)
    dummy_wave_rs = torchaudio.functional.resample(dummy_wave, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
    dummy_spec = soundstream.wav_to_spec(dummy_wave_rs.to(device))
    dummy_wave_rs2 = soundstream.spec_to_wav(dummy_spec)
    dummy_wave2 = torchaudio.functional.resample(dummy_wave_rs2,  ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)
    
    if dummy_wave.shape[-1] == dummy_wave2.shape[-1] and dummy_spec.shape[1] % audio_spec_count_vae == 0:
        print("pc ", pose_count)
        print("dummy_wave s ", dummy_wave.shape)
        print("dummy_spec s ", dummy_spec.shape)
        print("dummy_spec s / audio_spec_count_vae ", dummy_spec.shape[1] / audio_spec_count_vae)

"""

# determine number of spectra procuced by waveform of length audio_waveform_length_soundstream
audio_waveform = torch.zeros((1, audio_waveform_length_soundstream), dtype=torch.float32).to(device)
audio_waveform_rs = torchaudio.functional.resample(audio_waveform, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
audio_specs = soundstream.wav_to_spec(audio_waveform_rs)
audio_spec_count_soundstream = audio_specs.shape[1]
audio_spec_filter_count = audio_specs.shape[-1]
audio_waveform_rs_2 = soundstream.spec_to_wav(audio_specs)
audio_waveform_2 = torchaudio.functional.resample(audio_waveform_rs_2,  ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)

print("audio_spec_count_soundstream ", audio_spec_count_soundstream, " audio_spec_filter_count ", audio_spec_filter_count)


"""
Create Dataset
"""

class AudioDataset(Dataset):
    def __init__(self, audio_waveforms, audio_waveform_length, audio_data_count):
        self.audio_waveforms = audio_waveforms
        self.audio_waveform_length = audio_waveform_length
        self.audio_data_count = audio_data_count

    def __len__(self):
        return self.audio_data_count
    
    def __getitem__(self, idx):
        
        audio_index = torch.randint(0, len(self.audio_waveforms), size=(1,))
        audio_waveform = self.audio_waveforms[audio_index]
        
        audio_length = audio_waveform.shape[1]
        audio_excerpt_start = torch.randint(0, audio_length - self.audio_waveform_length, size=(1,))
        audio_excerpt = audio_waveform[:, audio_excerpt_start:audio_excerpt_start+self.audio_waveform_length]
        audio_excerpt = audio_excerpt[0]
        
        return audio_excerpt


full_dataset = AudioDataset(audio_all_data, audio_waveform_length_soundstream, data_count)
dataset_size = len(full_dataset)

data_item = full_dataset[0]

print("data_item s ", data_item.shape)

dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

batch_x = next(iter(dataloader))

print("batch_x s ", batch_x.shape)

# test conversion of audio waveform into soundstream spectra

audio_batch = next(iter(dataloader)).to(device)
audio_batch_rs =  torchaudio.functional.resample(audio_batch, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
audio_batch_specs = soundstream.wav_to_spec(audio_batch_rs)

print("audio_batch s ", audio_batch.shape)
print("audio_batch_rs s ", audio_batch_rs.shape)
print("audio_batch_specs s ", audio_batch_specs.shape)

"""
Create Models
"""

# create encoder model

class Encoder(nn.Module):
    
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

encoder = Encoder(latent_dim, audio_spec_count_vae, audio_spec_filter_count, vae_conv_channel_counts, vae_conv_kernel_size, vae_dense_layer_sizes).to(device)

print(encoder)

# test encoder
audio_batch = next(iter(dataloader)).to(device)
audio_batch_rs =  torchaudio.functional.resample(audio_batch, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
audio_batch_specs = soundstream.wav_to_spec(audio_batch_rs)

print("audio_batch s ", audio_batch.shape)
print("audio_batch_rs s ", audio_batch_rs.shape)
print("audio_batch_specs s ", audio_batch_specs.shape)

# regroup specs
# from: batch_size, audio_spec_count_soundstream, audio_spec_filter_count
# to: batch_size2, audio_spec_count_vae, audio_spec_filter_count with batch_size2 = batch_size * audio_spec_count_soundstream // audio_spec_count_vae
audio_batch_specs = audio_batch_specs.reshape((-1, audio_spec_count_vae, audio_spec_filter_count))
print("audio_batch_specs 2 s ", audio_batch_specs.shape)

# permute specs 
# from: batch_size2, audio_spec_count_vae, audio_spec_filter_count
# to: batch_size2, audio_spec_filter_count, audio_spec_count_vae
audio_batch_specs = audio_batch_specs.permute((0, 2, 1))
print("audio_batch_specs 2 s ", audio_batch_specs.shape)

# reshape specs
# from: batch_size2, audio_spec_filter_count, audio_spec_count_vae
# to: batch_size2, 1, audio_spec_filter_count, audio_spec_count_vae
audio_batch_specs = audio_batch_specs.unsqueeze(1)
print("audio_batch_specs 3 s ", audio_batch_specs.shape)

audio_encoder_in = audio_batch_specs
audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
audio_encoder_out = encoder.reparameterize(audio_encoder_out_mu, audio_encoder_out_std)

print("audio_batch s ", audio_batch.shape)
print("audio_batch_specs s ", audio_batch_specs.shape)
print("audio_encoder_in s ", audio_encoder_in.shape)
print("audio_encoder_out_mu s ", audio_encoder_out_mu.shape)
print("audio_encoder_out_std s ", audio_encoder_out_std.shape)
print("audio_encoder_out s ", audio_encoder_out.shape)

if load_weights and encoder_weights_file:
    encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))
    
# Decoder 
class Decoder(nn.Module):
    
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
    
vae_conv_channel_counts_reversed = vae_conv_channel_counts.copy()
vae_conv_channel_counts_reversed.reverse()
    
vae_dense_layer_sizes_reversed = vae_dense_layer_sizes.copy()
vae_dense_layer_sizes_reversed.reverse()

vae_conv_channel_counts_reversed

decoder = Decoder(latent_dim, audio_spec_count_vae, audio_spec_filter_count, vae_conv_channel_counts_reversed, vae_conv_kernel_size, vae_dense_layer_sizes_reversed).to(device)

print(decoder)

# test decoder
audio_decoder_in = audio_encoder_out
audio_decoder_out = decoder(audio_decoder_in)

# reshape specs
# from: batch_size2, 1, audio_spec_filter_count, audio_spec_count_vae
# to: batch_size2, audio_spec_filter_count, audio_spec_count_vae
audio_batch_specs = audio_decoder_out.squeeze(1)
print("audio_batch_specs s ", audio_batch_specs.shape)

# permute specs 
# from: batch_size2, audio_spec_filter_count, audio_spec_count_vae
# to: batch_size2, audio_spec_count_vae, audio_spec_filter_count
audio_batch_specs = audio_batch_specs.permute((0, 2, 1))
print("audio_batch_specs 2 s ", audio_batch_specs.shape)

# regroup specs
# from: batch_size2, audio_spec_count_vae, audio_spec_filter_count
# to: batch_size, audio_spec_count_soundstream, audio_spec_filter_count
audio_batch_specs = audio_batch_specs.reshape((batch_size, audio_spec_count_soundstream, audio_spec_filter_count))
print("audio_batch_specs 2 s ", audio_batch_specs.shape)

audio_batch_rs = soundstream.spec_to_wav(audio_batch_specs)
audio_batch =  torchaudio.functional.resample(audio_batch_rs, ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)

print("audio_decoder_in s ", audio_decoder_in.shape)
print("audio_decoder_out s ", audio_decoder_out.shape)
print("audio_batch_specs s ", audio_batch_specs.shape)
print("audio_batch_rs s ", audio_batch_rs.shape)
print("audio_batch s ", audio_batch.shape)

if load_weights and decoder_weights_file:
    decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))

"""
Training
"""

def calc_ae_beta_values():
    
    ae_beta_values = []

    for e in range(epochs):
        
        cycle_step = e % ae_beta_cycle_duration
        
        #print("cycle_step ", cycle_step)

        if cycle_step < ae_beta_min_const_duration:
            ae_beta_value = ae_min_beta
            ae_beta_values.append(ae_beta_value)
        elif cycle_step > ae_beta_cycle_duration - ae_beta_max_const_duration:
            ae_beta_value = ae_max_beta
            ae_beta_values.append(ae_beta_value)
        else:
            lin_step = cycle_step - ae_beta_min_const_duration
            ae_beta_value = ae_min_beta + (ae_max_beta - ae_min_beta) * lin_step / (ae_beta_cycle_duration - ae_beta_min_const_duration - ae_beta_max_const_duration)
            ae_beta_values.append(ae_beta_value)
            
    return ae_beta_values

ae_beta_values = calc_ae_beta_values()

ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=ae_learning_rate)
ae_scheduler = torch.optim.lr_scheduler.StepLR(ae_optimizer, step_size=50, gamma=0.316) # reduce the learning every 100 epochs by a factor of 10

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
   
def variational_loss2(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #alternative: mean squared distance from ideal mu=0 and std=1:
    vl=torch.mean(mu.pow(2)+(1-std).pow(2))
    return vl

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

def ae_spec_loss(y_specs, yhat_specs):
    
    _aml = mse_loss(yhat_specs, y_specs)

    return _aml

def ae_perc_loss(y_wave, yhat_wave):
    
    apl = perc_loss(yhat_wave, y_wave)
    
    return apl

def ae_loss(y_wave, yhat_wave, y_specs, yhat_specs, mu, std):

    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
    
    # ae mel_rec loss
    _ae_spec_rec_loss = ae_spec_loss(y_specs, yhat_specs)
    
    # ae_perc_rec_loss
    _ae_perc_rec_loss = ae_perc_loss(y_wave, yhat_wave)
    
    _total_loss = 0.0
    _total_loss += _ae_spec_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_perc_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_kld_loss * ae_beta
    
    return _total_loss, _ae_spec_rec_loss, _ae_perc_rec_loss, _ae_kld_loss

def ae_train_step(y_wave):
    
    #print("y_wave s ", y_wave.shape)
    
    batch_size = y_wave.shape[0]

    y_wave_rs =  torchaudio.functional.resample(y_wave, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
    
    #print("y_wave_rs s ", y_wave_rs.shape)
    
    y_specs = soundstream.wav_to_spec(y_wave_rs)

    #print("y_specs s ", y_specs.shape)
    
    # regroup specs
    # from: batch_size, audio_spec_count_soundstream, audio_spec_filter_count
    # to: batch_size2, audio_spec_count_vae, audio_spec_filter_count with batch_size2 = batch_size * audio_spec_count_soundstream // audio_spec_count_vae
    y_specs_regrouped = y_specs.reshape((-1, audio_spec_count_vae, audio_spec_filter_count))
    #print("y_specs_regrouped 2 s ", y_specs_regrouped.shape)

    # permute specs 
    # from: batch_size2, audio_spec_count_vae, audio_spec_filter_count
    # to: batch_size2, audio_spec_filter_count, audio_spec_count_vae
    y_specs_regrouped = y_specs_regrouped.permute((0, 2, 1))
    #print("y_specs_regrouped 2 s ", y_specs_regrouped.shape)
    
    # reshape specs
    # from: batch_size2, audio_spec_filter_count, audio_spec_count_vae
    # to: batch_size2, 1, audio_spec_filter_count, audio_spec_count_vae
    y_specs_regrouped = y_specs_regrouped.unsqueeze(1)
    #print("y_specs_regrouped 3 s ", y_specs_regrouped.shape)
    
    # encode mels 
    audio_encoder_out_mu, audio_encoder_out_std = encoder(y_specs_regrouped)
    
    #print("audio_encoder_out_mu s ", audio_encoder_out_mu.shape)
    
    mu = audio_encoder_out_mu
    std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-8
    decoder_input = encoder.reparameterize(mu, std)
    
    #print("decoder_input s ", decoder_input.shape)
    
    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
 
    yhat_specs_regrouped = decoder(decoder_input)
    
    #print("yhat_specs_regrouped s ", yhat_specs_regrouped.shape)
    
    # ae spec rec loss
    _ae_spec_rec_loss = ae_spec_loss(y_specs_regrouped, yhat_specs_regrouped)
    
    # convert the spec grouping back to original
    
    # reshape specs
    # from: batch_size2, 1, audio_spec_filter_count, audio_spec_count_vae
    # to: batch_size2, audio_spec_filter_count, audio_spec_count_vae
    yhat_specs_regrouped = yhat_specs_regrouped.squeeze(1)
    #print("yhat_specs_regrouped s ", yhat_specs_regrouped.shape)
    
    # permute specs 
    # from: batch_size2, audio_spec_filter_count, audio_spec_count_vae
    # to: batch_size2, audio_spec_count_vae, audio_spec_filter_count
    yhat_specs_regrouped = yhat_specs_regrouped.permute((0, 2, 1))
    #print("yhat_specs_regrouped 2 s ", yhat_specs_regrouped.shape)
    
    # regroup specs
    # from: batch_size2, audio_spec_count_vae, audio_spec_filter_count
    # to: batch_size, audio_spec_count_soundstream, audio_spec_filter_count
    yhat_specs_regrouped = yhat_specs_regrouped.reshape((batch_size, audio_spec_count_soundstream, audio_spec_filter_count))
    #print("yhat_specs_regrouped 2 s ", yhat_specs_regrouped.shape)
        
    yhat_wave_rs = soundstream.spec_to_wav(yhat_specs_regrouped)
    
    #print("yhat_wave_rs s ", yhat_wave_rs.shape)
    
    yhat_wave =  torchaudio.functional.resample(yhat_wave_rs, ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)

    #print("yhat_wave s ", yhat_wave.shape)


    # ae perc rec loss
    _ae_perc_rec_loss = ae_perc_loss(y_wave, yhat_wave.unsqueeze(1))

    _total_loss = 0.0
    _total_loss += _ae_spec_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_perc_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_kld_loss * ae_beta

    # Backpropagation
    ae_optimizer.zero_grad()
    _total_loss.backward()
    
    #torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.01)
    #torch.nn.utils.clip_grad_norm(decoder.parameters(), 0.01)

    ae_optimizer.step()
    
    return _total_loss, _ae_spec_rec_loss, _ae_perc_rec_loss, _ae_kld_loss

"""
# ae_train_step

audio_batch = next(iter(dataloader)).to(device)
test_loss = ae_train_step(audio_batch)
"""

def train(dataloader, epochs):
    
    global ae_beta
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae spec"] = []
    loss_history["ae perc"] = []
    loss_history["ae kld"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_beta = ae_beta_values[epoch]
        
        #print("ae_kld_loss_scale ", ae_kld_loss_scale)
        
        ae_train_loss_per_epoch = []
        ae_spec_loss_per_epoch = []
        ae_perc_loss_per_epoch = []
        ae_kld_loss_per_epoch = []
        
        for train_batch in dataloader:
            train_batch = train_batch.to(device)
            
            _ae_loss, _ae_spec_loss, _ae_perc_loss, _ae_kld_loss = ae_train_step(train_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_spec_loss = _ae_spec_loss.detach().cpu().numpy()
            _ae_perc_loss = _ae_perc_loss.detach().cpu().numpy()
            _ae_kld_loss = _ae_kld_loss.detach().cpu().numpy()
            
            #print("_ae_prior_loss ", _ae_prior_loss)
            
            ae_train_loss_per_epoch.append(_ae_loss)
            ae_spec_loss_per_epoch.append(_ae_spec_loss)
            ae_perc_loss_per_epoch.append(_ae_perc_loss)
            ae_kld_loss_per_epoch.append(_ae_kld_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_spec_loss_per_epoch = np.mean(np.array(ae_spec_loss_per_epoch))
        ae_perc_loss_per_epoch = np.mean(np.array(ae_perc_loss_per_epoch))
        ae_kld_loss_per_epoch = np.mean(np.array(ae_kld_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epoch))
            torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epoch))
        
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae spec"].append(ae_spec_loss_per_epoch)
        loss_history["ae perc"].append(ae_perc_loss_per_epoch)
        loss_history["ae kld"].append(ae_kld_loss_per_epoch)
        
        print ('epoch {} : ae train: {:01.4f} spec {:01.4f} perc {:01.4f} kld {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_spec_loss_per_epoch, ae_perc_loss_per_epoch, ae_kld_loss_per_epoch, time.time()-start))
    
        ae_scheduler.step()
        
    return loss_history

# fit model
loss_history = train(dataloader, epochs)

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
torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epochs))
torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epochs))


"""
Inference
"""

def create_ref_audio_window(waveform_window, file_name):

    torchaudio.save("{}".format(file_name), waveform_window, audio_orig_sample_rate)

@torch.no_grad()
def create_soundstream_audio_window(waveform_window, file_name):

    with torch.no_grad():
        
        audio_wave_rs = torchaudio.functional.resample(waveform_window, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
        audio_spec = soundstream.wav_to_spec(audio_wave_rs.to(device))
        audio_wave_rs2 = soundstream.spec_to_wav(audio_spec)
        audio_wave2 = torchaudio.functional.resample(audio_wave_rs2,  ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)

    torchaudio.save("{}".format(file_name), audio_wave2.detach().cpu(), audio_orig_sample_rate)

@torch.no_grad()
def create_pred_audio_window(waveform_window, file_name):
    
    batch_size = 1
    
    encoder.eval()
    decoder.eval()
    
    y_wave = waveform_window.to(device)
    #print("y_wave s ", y_wave.shape)
    
    y_wave_rs = torchaudio.functional.resample(y_wave, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
    #print("y_wave_rs s ", y_wave_rs.shape)
    
    y_specs = soundstream.wav_to_spec(y_wave_rs.to(device))
    #print("y_specs s ", y_specs.shape)
    
    # regroup specs
    y_specs_regrouped = y_specs.reshape((-1, audio_spec_count_vae, audio_spec_filter_count))
    #print("y_specs_regrouped s ", y_specs_regrouped.shape)
    
    # permute specs 
    y_specs_regrouped = y_specs_regrouped.permute((0, 2, 1))
    #print("y_specs_regrouped 2 s ", y_specs_regrouped.shape)
    
    # reshape specs
    y_specs_regrouped = y_specs_regrouped.unsqueeze(1)
    #print("y_specs_regrouped 3 s ", y_specs_regrouped.shape)
    
    # encode mels 
    audio_encoder_out_mu, audio_encoder_out_std = encoder(y_specs_regrouped)
    
    #print("audio_encoder_out_mu s ", audio_encoder_out_mu.shape)
    
    mu = audio_encoder_out_mu
    std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-8
    decoder_input = encoder.reparameterize(mu, std)
    
    #print("decoder_input s ", decoder_input.shape)
    
    yhat_specs_regrouped = decoder(decoder_input)
    #print("yhat_specs_regrouped s ", yhat_specs_regrouped.shape)
    
    # reshape specs
    yhat_specs_regrouped = yhat_specs_regrouped.squeeze(1)
    #print("yhat_specs_regrouped s ", yhat_specs_regrouped.shape)
    
    # permute specs 
    yhat_specs_regrouped = yhat_specs_regrouped.permute((0, 2, 1))
    #print("yhat_specs_regrouped 2 s ", yhat_specs_regrouped.shape)
    
    # regroup specs
    yhat_specs = yhat_specs_regrouped.reshape((batch_size, audio_spec_count_soundstream, audio_spec_filter_count))
    #print("yhat_specs 3 s ", yhat_specs.shape)
        
    yhat_wave_rs = soundstream.spec_to_wav(yhat_specs)
    #print("yhat_wave_rs s ", yhat_wave_rs.shape)
    
    yhat_wave =  torchaudio.functional.resample(yhat_wave_rs, ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)
    #print("yhat_wave s ", yhat_wave.shape)
        
    torchaudio.save("{}".format(file_name), yhat_wave.detach().cpu(), audio_orig_sample_rate)

    encoder.train()
    decoder.train()
    
test_waveform, _ = torchaudio.load("E:/Data/audio/Eleni/4_5870821179501060412.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_48khz.wav")
test_waveform_sample_index = audio_orig_sample_rate * 10
test_waveform_window = test_waveform[:, test_waveform_sample_index:test_waveform_sample_index+audio_waveform_length_soundstream]

create_ref_audio_window(test_waveform_window, "results/audio/audio_window_orig.wav")
create_soundstream_audio_window(test_waveform_window, "results/audio/audio_window_soundstream.wav")
create_pred_audio_window(test_waveform_window, "results/audio/audio_window_pred_epoch_{}.wav".format(epochs))

def create_ref_audio(waveform, file_name):

    torchaudio.save("{}".format(file_name), waveform, audio_orig_sample_rate)
    
    #print("waveform s ", waveform.shape)

@torch.no_grad()
def create_soundstream_audio(waveform, file_name):
    
    waveform_length = waveform.shape[1]
    audio_window_offset = audio_waveform_length_soundstream // 2
    audio_window_env = torch.hann_window(audio_waveform_length_soundstream)
    
    audio_window_count = int(waveform_length - audio_waveform_length_soundstream) // audio_window_offset
    pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
    
    #print("pred_audio_sequence s ", pred_audio_sequence.shape)
    
    for i in range(audio_window_count):
        
        #print("i ", i)
        
        window_start = i * audio_window_offset
        window_end = window_start + audio_waveform_length_soundstream
        
        audio_wave = waveform[:, window_start:window_end].to(device)
        
        #print("audio_wave s ", audio_wave.shape)
    
        audio_wave_rs = torchaudio.functional.resample(audio_wave, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)

        #print("audio_wave_rs s ", audio_wave_rs.shape)
        
        audio_specs = soundstream.wav_to_spec(audio_wave_rs.to(device))
    
        #print("audio_specs s ", audio_specs.shape)
        
        audio_wave_rs2 = soundstream.spec_to_wav(audio_specs)
        #print("audio_wave_rs2 s ", audio_wave_rs2.shape)
        
        audio_wave2 =  torchaudio.functional.resample(audio_wave_rs2, ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)
        #print("audio_wave2 s ", audio_wave2.shape)
        
        audio_wave2 = audio_wave2.detach().cpu()

        pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_waveform_length_soundstream] += audio_wave2[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_orig_sample_rate)

@torch.no_grad()
def create_pred_audio(waveform, file_name):
    
    batch_size = 1
    
    encoder.eval()
    decoder.eval()
    
    waveform_length = waveform.shape[1]
    audio_window_offset = audio_waveform_length_soundstream // 2
    audio_window_env = torch.hann_window(audio_waveform_length_soundstream)
    
    audio_window_count = int(waveform_length - audio_waveform_length_soundstream) // audio_window_offset
    pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
    
    #print("pred_audio_sequence s ", pred_audio_sequence.shape)
    
    for i in range(audio_window_count):
        
        #print("i ", i)
        
        window_start = i * audio_window_offset
        window_end = window_start + audio_waveform_length_soundstream
        
        y_wave = waveform[:, window_start:window_end].to(device)
        
        #print("y_wave s ", y_wave.shape)
    
        y_wave_rs = torchaudio.functional.resample(y_wave, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)

        #print("y_wave_rs s ", y_wave_rs.shape)
        
        y_specs = soundstream.wav_to_spec(y_wave_rs.to(device))
    
        #print("y_specs s ", y_specs.shape)
        
        # regroup specs
        y_specs_regrouped = y_specs.reshape((-1, audio_spec_count_vae, audio_spec_filter_count))
        #print("y_specs_regrouped s ", y_specs_regrouped.shape)
        
        # permute specs 
        y_specs_regrouped = y_specs_regrouped.permute((0, 2, 1))
        #print("y_specs_regrouped 2 s ", y_specs_regrouped.shape)
        
        # reshape specs
        y_specs_regrouped = y_specs_regrouped.unsqueeze(1)
        #print("y_specs_regrouped 3 s ", y_specs_regrouped.shape)
        
        # encode mels 
        audio_encoder_out_mu, audio_encoder_out_std = encoder(y_specs_regrouped)
        
        #print("audio_encoder_out_mu s ", audio_encoder_out_mu.shape)
        
        mu = audio_encoder_out_mu
        std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-8
        decoder_input = encoder.reparameterize(mu, std)
        
        #print("decoder_input s ", decoder_input.shape)
        
        yhat_specs_regrouped = decoder(decoder_input)
        #print("yhat_specs_regrouped s ", yhat_specs_regrouped.shape)
        
        # reshape specs
        yhat_specs_regrouped = yhat_specs_regrouped.squeeze(1)
        #print("yhat_specs_regrouped s ", yhat_specs_regrouped.shape)
        
        # permute specs 
        yhat_specs_regrouped = yhat_specs_regrouped.permute((0, 2, 1))
        #print("yhat_specs_regrouped 2 s ", yhat_specs_regrouped.shape)
        
        # regroup specs
        yhat_specs = yhat_specs_regrouped.reshape((batch_size, audio_spec_count_soundstream, audio_spec_filter_count))
        #print("yhat_specs s ", yhat_specs.shape)
            
        yhat_wave_rs = soundstream.spec_to_wav(yhat_specs)
        #print("yhat_wave_rs s ", yhat_wave_rs.shape)
        
        yhat_wave =  torchaudio.functional.resample(yhat_wave_rs, ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)
        #print("yhat_wave s ", yhat_wave.shape)
        
        yhat_wave = yhat_wave.detach().cpu()

        pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_waveform_length_soundstream] += yhat_wave[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_orig_sample_rate)

    encoder.train()
    decoder.train()


#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take1__double_Bind_HQ_audio_crop_48khz.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take1__double_Bind_HQ_audio_crop_48khz.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take2_Hibr_II_HQ_audio_crop_48khz.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_48khz.wav")

test_waveform, _ = torchaudio.load("E:/Data/audio/Eleni/4_5870821179501060412.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_48khz.wav")

test_start_times_sec = [ 20, 120, 240 ]
test_duration_sec = 20

for test_start_time_sec in test_start_times_sec:
    start_time_frames = test_start_time_sec * audio_orig_sample_rate
    end_time_frames = start_time_frames + test_duration_sec * audio_orig_sample_rate

    create_ref_audio(test_waveform[:, start_time_frames:end_time_frames], "results/audio/audio_ref_{}-{}.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec)))
    create_soundstream_audio(test_waveform[:, start_time_frames:end_time_frames], "results/audio/audio_soundstream_{}-{}.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec)))
    create_pred_audio(test_waveform[:, start_time_frames:end_time_frames], "results/audio/audio_pred_{}-{}_epoch_{}_.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec), epochs))

@torch.no_grad()
def encode_audio(waveform):
    
    encoder.eval()
    
    waveform_length = waveform.shape[1]
    audio_window_offset = audio_waveform_length_soundstream // 2
    audio_window_env = torch.hann_window(audio_waveform_length_soundstream)
    
    audio_window_count = int(waveform_length - audio_waveform_length_soundstream) // audio_window_offset
    
    #print("audio_window_count ", audio_window_count)

    latent_vectors = []
    
    for i in range(audio_window_count):
    
        #print("i ", i)
        
        window_start = i * audio_window_offset
        window_end = window_start + audio_waveform_length_soundstream
        
        y_wave = waveform[:, window_start:window_end].to(device)
        
        #print("y_wave s ", y_wave.shape)
    
        y_wave_rs = torchaudio.functional.resample(y_wave, audio_orig_sample_rate, ssu.soundstream_audio_sample_rate)
    
        #print("y_wave_rs s ", y_wave_rs.shape)
        
        y_specs = soundstream.wav_to_spec(y_wave_rs.to(device))
    
        #print("y_specs s ", y_specs.shape)
        
        # regroup specs
        y_specs_regrouped = y_specs.reshape((-1, audio_spec_count_vae, audio_spec_filter_count))
        #print("y_specs_regrouped s ", y_specs_regrouped.shape)
        
        # permute specs 
        y_specs_regrouped = y_specs_regrouped.permute((0, 2, 1))
        #print("y_specs_regrouped 2 s ", y_specs_regrouped.shape)
        
        # reshape specs
        y_specs_regrouped = y_specs_regrouped.unsqueeze(1)
        #print("y_specs_regrouped 3 s ", y_specs_regrouped.shape)
        
        # encode mels 
        audio_encoder_out_mu, audio_encoder_out_std = encoder(y_specs_regrouped)
        
        #print("audio_encoder_out_mu s ", audio_encoder_out_mu.shape)
        
        mu = audio_encoder_out_mu
        std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-8
        decoder_input = encoder.reparameterize(mu, std)
        
        #print("decoder_input s ", decoder_input.shape)
    
        latent_vector = decoder_input.detach().cpu().numpy()
        
        #print("latent_vector s ", latent_vector.shape)
        
        latent_vectors.append(latent_vector)
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    encoder.train()
    
    return latent_vectors
  
@torch.no_grad()
def decode_audio_encodings(encodings, file_name):
    
    batch_size = 1
    
    decoder.eval()

    audio_window_offset = audio_waveform_length_soundstream // 2
    audio_window_env = torch.hann_window(audio_waveform_length_soundstream)
    
    audio_encoding_offset = audio_spec_count_soundstream // audio_spec_count_vae
    
    #print("audio_encoding_offset ", audio_encoding_offset)

    audio_window_count = encodings.shape[0] // audio_encoding_offset
    
    #print("audio_window_count ", audio_window_count)
    
    waveform_length = audio_window_count * audio_window_offset + audio_waveform_length_soundstream
    
    #print("waveform_length ", waveform_length)
    
    pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
    
    for i in range(audio_window_count):
        
        print("i ", i)
        
        decoder_input = encodings[i * audio_encoding_offset:(i + 1) * audio_encoding_offset, ... ]
        decoder_input = torch.from_numpy(decoder_input).to(device)
        
        #print("decoder_input s ", decoder_input.shape)
        
        yhat_specs_regrouped = decoder(decoder_input)
        #print("yhat_specs_regrouped s ", yhat_specs_regrouped.shape)
        
        # reshape specs
        yhat_specs_regrouped = yhat_specs_regrouped.squeeze(1)
        #print("yhat_specs_regrouped s ", yhat_specs_regrouped.shape)
        
        # permute specs 
        yhat_specs_regrouped = yhat_specs_regrouped.permute((0, 2, 1))
        #print("yhat_specs_regrouped 2 s ", yhat_specs_regrouped.shape)
        
        # regroup specs
        yhat_specs = yhat_specs_regrouped.reshape((batch_size, audio_spec_count_soundstream, audio_spec_filter_count))
        #print("yhat_specs s ", yhat_specs.shape)
            
        yhat_wave_rs = soundstream.spec_to_wav(yhat_specs)
        #print("yhat_wave_rs s ", yhat_wave_rs.shape)
        
        yhat_wave =  torchaudio.functional.resample(yhat_wave_rs, ssu.soundstream_audio_sample_rate, audio_orig_sample_rate)
        #print("yhat_wave s ", yhat_wave.shape)
        
        yhat_wave = yhat_wave.detach().cpu()

        pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_waveform_length_soundstream] += yhat_wave[0] * audio_window_env
    
    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_orig_sample_rate)

    decoder.train()


# reconstruct original waveform

test_start_time_sec = 20
test_duration_sec = 20
start_time_frames = test_start_time_sec * audio_orig_sample_rate
end_time_frames = start_time_frames + test_duration_sec * audio_orig_sample_rate
    
latent_vectors = encode_audio(test_waveform[:, start_time_frames:end_time_frames])
decode_audio_encodings(latent_vectors, "results/audio/rec_audio_epochs_{}_audio_{}-{}.wav".format(epochs, test_start_time_sec, test_duration_sec))

# random walk
# somewhat broken, needs fixing

test_start_time_sec = 20
test_duration_sec = 20

start_time_frames = test_start_time_sec * audio_orig_sample_rate
end_time_frames = start_time_frames + test_duration_sec * audio_orig_sample_rate

latent_vectors = encode_audio(test_waveform[:, start_time_frames:end_time_frames])
latent_vector_count = latent_vectors.shape[0]
latent_vectors = [ latent_vectors[:1] ]

for lvI in range(latent_vector_count - 1):
    random_step = np.random.random((latent_dim)).astype(np.float32) * 0.1
    latent_vectors.append(latent_vectors[lvI] + random_step)

latent_vectors = np.concatenate(latent_vectors, axis=0)

decode_audio_encodings(latent_vectors, "results/audio/randwalk_audio_epochs_{}_audio_{}-{}.wav".format(epochs, test_start_time_sec, test_duration_sec))


# sequence offset following

test_start_time_sec = 20
test_duration_sec = 20

start_time_frames = test_start_time_sec * audio_orig_sample_rate
end_time_frames = start_time_frames + test_duration_sec * audio_orig_sample_rate

latent_vectors = encode_audio(test_waveform[:, start_time_frames:end_time_frames])

offset_encodings = []

for lvI in range(latent_vectors.shape[0]):
    sin_value = np.sin(lvI / (len(latent_vectors) - 1) * np.pi * 4.0)
    offset = np.ones(shape=(latent_dim), dtype=np.float32) * sin_value * 1.0
    offset_encoding = latent_vectors[lvI] + offset
    offset_encodings.append(offset_encoding)

offset_encodings = np.stack(offset_encodings, axis=0)

decode_audio_encodings(offset_encodings, "results/audio/offset_audio_epochs_{}_audio_{}-{}.wav".format(epochs, test_start_time_sec, test_duration_sec))


# interpolate two original sequences

test1_start_time_sec = 20
test2_start_time_sec = 120
test_duration_sec = 20

start1_time_frames = test1_start_time_sec * audio_orig_sample_rate
end1_time_frames = start1_time_frames + test_duration_sec * audio_orig_sample_rate

start2_time_frames = test2_start_time_sec * audio_orig_sample_rate
end2_time_frames = start2_time_frames + test_duration_sec * audio_orig_sample_rate

latent_vectors_1 = encode_audio(test_waveform[:, start1_time_frames:end1_time_frames])
latent_vectors_2 = encode_audio(test_waveform[:, start2_time_frames:end2_time_frames])

mix_encodings = []

for lvI in range(latent_vectors_1.shape[0]):
    mix_factor = lvI / (latent_vectors_1.shape[0] - 1)
    mix_encoding = latent_vectors_1[lvI] * (1.0 - mix_factor) + latent_vectors_2[lvI] * mix_factor
    mix_encodings.append(mix_encoding)

mix_encodings = np.stack(mix_encodings, axis=0)

decode_audio_encodings(mix_encodings, "results/audio/mix_audio_epochs_{}_audio1_{}-{}_audio2_{}-{}.wav".format(epochs, test1_start_time_sec, test1_start_time_sec + test_duration_sec, test2_start_time_sec, test2_start_time_sec + test_duration_sec))



