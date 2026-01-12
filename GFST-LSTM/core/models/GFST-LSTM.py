import math
import torch
import torch.nn as nn
from core.layers.ST_LSTMCell import SpatioTemporalLSTMCell
import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class FourierConv(nn.Module):
    def __init__(self, channels):
        super(FourierConv, self).__init__()
        self.channels = channels
        self.weight_real = nn.Parameter(torch.ones(channels))
        self.weight_imag = nn.Parameter(torch.zeros(channels))
        
    def forward(self, x):
        B, C, H, W = x.size()
        assert C == self.channels, f"Input channels {C} does not match expected channels {self.channels}"
        x_fft = torch.fft.rfft2(x, norm="ortho")
        weight_complex = torch.complex(self.weight_real, self.weight_imag)
        weight_complex = weight_complex.view(1, C, 1, 1).to(x_fft.device)
        x_fft_weighted = x_fft * weight_complex
        x_reconstructed = torch.fft.irfft2(x_fft_weighted, s=(H, W), norm="ortho")
        
        return x_reconstructed

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel  # 1*1*1
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.sr_size = 8
        
        cell_list = []
        width = configs.img_width // configs.patch_size // self.sr_size
        height = configs.img_height // configs.patch_size // self.sr_size
        
        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                        configs.stride)
            )
        
        self.cell_list = nn.ModuleList(cell_list)
        
        self.merge = nn.Sequential(
            nn.Conv2d(self.frame_channel, self.num_hidden[-1], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        
        self.Downsample = nn.Sequential(
            nn.Conv2d(self.num_hidden[0], self.num_hidden[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(self.num_hidden[-1], self.num_hidden[-1], kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        
        self.CBAM = CBAM(in_channels=64)
        self.ffc_modules = nn.ModuleList()
        if num_layers > 1:
            for i in range(1, num_layers):
                self.ffc_modules.append(FourierConv(num_hidden[i-1]))

    def forward(self, frames, mask_true):
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        height = frames.shape[3] // self.sr_size  # 64/4=16
        width = frames.shape[4] // self.sr_size
        n = int(math.log2(self.sr_size))
        
        next_frames = []
        h_t = []
        c_t = []
        x_gen = None
        
        for i in range(self.num_layers):
            zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            
        memory = torch.zeros([batch_size, self.num_hidden[0], height, width]).to(self.configs.device)
        
        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]  # (4,1,64,64)
            else:
                time_diff = t - self.configs.input_length
                net = x_gen  
                
            frames_feature = net
            x_t = self.merge(frames_feature)
            x_t = self.CBAM(x_t)
            
            for i in range(n):
                x_t = self.Downsample(x_t)
                
            h_t[0], c_t[0], memory = self.cell_list[0](x_t, h_t[0], c_t[0], memory)
            
            for i in range(1, self.num_layers):
                h_prev_transformed = self.ffc_modules[i-1](h_t[i-1])
                
                h_t[i], c_t[i], memory = self.cell_list[i](h_prev_transformed, h_t[i], c_t[i], memory)
            
            out = h_t[self.num_layers - 1]
            
            for i in range(n):
                out = self.Upsample(out)
                
            x_gen = self.srcnn(out)
            next_frames.append(x_gen)
            
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
