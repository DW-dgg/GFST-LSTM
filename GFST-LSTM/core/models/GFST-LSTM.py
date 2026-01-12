import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierConv(nn.Module):
    """
    Fast Fourier Convolution (FFC) module
    根据文章描述，将输入分为局部分支(60%)和全局分支(40%)
    """
    def __init__(self, channels):
        super(FourierConv, self).__init__()
        self.channels = channels
        
        # 按照60%-40%比例分割通道
        self.local_channels = int(channels * 0.6)
        self.global_channels = channels - self.local_channels
        
        # 局部分支：5x5卷积提取细粒度空间特征
        self.local_conv = nn.Conv2d(
            self.local_channels, 
            self.local_channels, 
            kernel_size=5, 
            padding=2,
            bias=False
        )
        self.local_bn = nn.BatchNorm2d(self.local_channels)
        
        # 全局分支：频域复值滤波器
        self.weight_real = nn.Parameter(torch.ones(self.global_channels))
        self.weight_imag = nn.Parameter(torch.zeros(self.global_channels))
        
        # 初始化时强调低频分量
        with torch.no_grad():
            self.weight_real.data *= 1.5
        
    def forward(self, x):
        B, C, H, W = x.size()
        assert C == self.channels, f"Input channels {C} does not match expected channels {self.channels}"
        
        # 按通道分割输入为局部分支和全局分支
        x_local = x[:, :self.local_channels, :, :]
        x_global = x[:, self.local_channels:, :, :]
        
        # ===== 局部分支：5x5卷积 =====
        out_local = self.local_conv(x_local)
        out_local = self.local_bn(out_local)
        out_local = F.relu(out_local)
        
        # ===== 全局分支：FFT -> 频域滤波 -> IFFT =====
        x_fft = torch.fft.rfft2(x_global, norm="ortho")
        
        weight_complex = torch.complex(self.weight_real, self.weight_imag)
        weight_complex = weight_complex.view(1, self.global_channels, 1, 1).to(x_fft.device)
        
        x_fft_weighted = x_fft * weight_complex
        out_global = torch.fft.irfft2(x_fft_weighted, s=(H, W), norm="ortho")
        
        # ===== 拼接两个分支的输出 =====
        out = torch.cat([out_local, out_global], dim=1)
        
        return out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attention
        
        return x


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(SpatioTemporalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        self.conv_x = nn.Conv2d(
            in_channel, num_hidden * 7, 
            kernel_size=filter_size, 
            stride=stride, 
            padding=self.padding
        )
        self.conv_h = nn.Conv2d(
            num_hidden, num_hidden * 4, 
            kernel_size=filter_size, 
            stride=stride, 
            padding=self.padding
        )
        self.conv_m = nn.Conv2d(
            num_hidden, num_hidden * 3, 
            kernel_size=filter_size, 
            stride=stride, 
            padding=self.padding
        )
        self.conv_c = nn.Conv2d(
            num_hidden, num_hidden * 2, 
            kernel_size=filter_size, 
            stride=stride, 
            padding=self.padding
        )
        
    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        c_concat = self.conv_c(c_t)
        
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_c, f_c = torch.split(c_concat, self.num_hidden, dim=1)
        
        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        c_new = f_t * c_t + i_t * torch.tanh(g_x + g_h)
        
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        m_new = f_t_prime * m_t + i_t_prime * torch.tanh(g_x_prime + g_m)
        
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new + m_new)
        
        return h_new, c_new, m_new


class GFST_LSTM(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(GFST_LSTM, self).__init__()
        
        self.configs = configs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        
        self.merge = nn.Conv2d(
            configs.patch_size * configs.patch_size * configs.img_channel,
            num_hidden[0],
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.CBAM = CBAM(num_hidden[0])
        
        self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.srcnn = nn.Sequential(
            nn.Conv2d(num_hidden[-1], 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, configs.patch_size * configs.patch_size * configs.img_channel, 
                     kernel_size=5, padding=2)
        )
        
        cell_list = []
        height = configs.img_width // configs.patch_size
        width = configs.img_width // configs.patch_size
        
        n = configs.sampling_times
        for _ in range(n):
            height = height // 2
            width = width // 2
            
        for i in range(num_layers):
            in_channel = num_hidden[i-1] if i > 0 else num_hidden[0]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel, num_hidden[i], height, width,
                    configs.filter_size, configs.stride
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        
        self.ffc_modules = nn.ModuleList()
        if num_layers > 1:
            for i in range(1, num_layers):
                self.ffc_modules.append(FourierConv(num_hidden[i-1]))
        
    def forward(self, frames, mask_true):
        batch = frames.shape[0]
        height = frames.shape[2]
        width = frames.shape[3]
        
        next_frames = []
        h_t = []
        c_t = []
        
        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], 
                 height // self.configs.patch_size // (2 ** self.configs.sampling_times),
                 width // self.configs.patch_size // (2 ** self.configs.sampling_times)]
            ).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            
        memory = torch.zeros(
            [batch, self.num_hidden[0],
             height // self.configs.patch_size // (2 ** self.configs.sampling_times),
             width // self.configs.patch_size // (2 ** self.configs.sampling_times)]
        ).to(self.configs.device)
        
        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            
            frames_feature = net
            x_t = self.merge(frames_feature)
            x_t = self.CBAM(x_t)
            
            n = self.configs.sampling_times
            for i in range(n):
                x_t = self.Downsample(x_t)
            
            h_t[0], c_t[0], memory = self.cell_list[0](x_t, h_t[0], c_t[0], memory)
            
            for i in range(1, self.num_layers):
                h_prev_transformed = self.ffc_modules[i-1](h_t[i-1])
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_prev_transformed, h_t[i], c_t[i], memory
                )
            
            out = h_t[self.num_layers - 1]
            
            for i in range(n):
                out = self.Upsample(out)
            
            x_gen = self.srcnn(out)
            next_frames.append(x_gen)
        
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        
        return next_frames
