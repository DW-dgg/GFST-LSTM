import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierConv(nn.Module):
    """
    Fast Fourier Convolution (FFC) module
    修改：更精确的频域滤波器初始化
    """
    def __init__(self, channels, h, w):
        super(FourierConv, self).__init__()
        self.channels = channels
        
        self.local_channels = int(channels * 0.6)
        self.global_channels = channels - self.local_channels
        
        # 本地分支：空域卷积
        self.local_conv = nn.Conv2d(
            self.local_channels, 
            self.local_channels, 
            kernel_size=5, 
            padding=2,
            bias=False
        )
        self.local_bn = nn.BatchNorm2d(self.local_channels)
        
        # 频域滤波器 W_freq：扩展为 3D Tensor [C_global, H, W//2+1]
        self.h_fft = h
        self.w_fft = w // 2 + 1
        self.weight_real = nn.Parameter(torch.empty(self.global_channels, self.h_fft, self.w_fft))
        self.weight_imag = nn.Parameter(torch.empty(self.global_channels, self.h_fft, self.w_fft))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        更精确的频域滤波器初始化：考虑FFT的对称性
        低频分量位于四个角落（对于rfft2，主要在左侧边界）
        """
        with torch.no_grad():
            for i in range(self.h_fft):
                for j in range(self.w_fft):
                    # 计算频率距离（考虑周期性边界条件）
                    # 对于高度维度，频率从中心向两边递增
                    freq_h = min(i, self.h_fft - i)
                    # 对于宽度维度（rfft2），频率从左边界递增
                    freq_w = j
                    
                    # 归一化的频率距离
                    freq_dist = ((freq_h / (self.h_fft / 2))**2 + 
                                (freq_w / self.w_fft)**2)**0.5
                    
                    # 指数衰减初始化：低频权重大，高频权重小
                    # 使用更陡峭的衰减以更强调低频
                    val = 1.5 * torch.exp(torch.tensor(-2.0 * freq_dist))
                    self.weight_real[:, i, j] = val
            
            # 虚部初始化为 0
            self.weight_imag.fill_(0.0)
        
    def forward(self, x):
        B, C, H, W = x.size()
        assert C == self.channels, f"Input channels {C} does not match expected channels {self.channels}"
        
        # 分割为本地和全局分支
        x_local = x[:, :self.local_channels, :, :]
        x_global = x[:, self.local_channels:, :, :]
        
        # 本地分支：空域卷积
        out_local = self.local_conv(x_local)
        out_local = self.local_bn(out_local)
        out_local = F.relu(out_local)

        # 全局分支：频域处理
        x_fft = torch.fft.rfft2(x_global, norm="ortho")
        
        # 应用可学习的频域滤波器 W_freq
        weight_complex = torch.complex(self.weight_real, self.weight_imag).to(x_fft.device)
        x_fft_weighted = x_fft * weight_complex
        
        # 逆傅里叶变换回空域
        out_global = torch.fft.irfft2(x_fft_weighted, s=(H, W), norm="ortho")

        # 合并本地和全局分支
        out = torch.cat([out_local, out_global], dim=1)
        
        return out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module with 3D Permutation
    新增：3D置换操作，将C×H×W转换为H×W×C
    """
    def __init__(self, channels, bottleneck_d=256):
        super(CBAM, self).__init__()
        
        # 通道注意力子模块
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP 架构：输入 -> 瓶颈层(d=256) -> 输出
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, bottleneck_d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(bottleneck_d, channels, 1, bias=False)
        )
        
        # 空间注意力子模块
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 3D置换操作：C×H×W -> H×W×C
        B, C, H, W = x.shape
        x_permuted = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # 通过MLP处理（需要转回C×H×W格式）
        x = x_permuted.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 通道注意力 Mc
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention
        
        # 空间注意力 Ms
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attention
        
        return x


class SpatioTemporalLSTMCell(nn.Module):
    """
    Spatio-Temporal LSTM Cell
    保持原有实现不变
    """
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
    """
    GFST-LSTM 网络主体结构
    修改：
    1. 使用卷积和转置卷积进行下采样/上采样
    2. 输出层改为简单的1×1卷积
    3. 集成3D置换操作
    """
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
        
        self.CBAM = CBAM(num_hidden[0], bottleneck_d=256)
        
        # 修改1：使用卷积进行下采样（替代最大池化）
        self.downsample_layers = nn.ModuleList()
        n = configs.sampling_times
        for i in range(n):
            self.downsample_layers.append(
                nn.Conv2d(
                    num_hidden[0], 
                    num_hidden[0], 
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                    bias=False
                )
            )
        
        # 修改1：使用转置卷积进行上采样（替代双线性插值）
        self.upsample_layers = nn.ModuleList()
        for i in range(n):
            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    num_hidden[-1], 
                    num_hidden[-1], 
                    kernel_size=4, 
                    stride=2, 
                    padding=1,
                    bias=False
                )
            )
        
        # 修改2：输出层改为简单的1×1卷积（替代SRCNN）
        self.output_conv = nn.Conv2d(
            num_hidden[-1], 
            configs.patch_size * configs.patch_size * configs.img_channel,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # 计算特征图尺寸
        cell_list = []
        height = configs.img_width // configs.patch_size
        width = configs.img_width // configs.patch_size
        
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
        
        # FFC 模块
        self.ffc_modules = nn.ModuleList()
        if num_layers > 1:
            for i in range(1, num_layers):
                self.ffc_modules.append(
                    FourierConv(num_hidden[i-1], h=height, w=width)
                )
        
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
            
            # 修改3：应用CBAM（包含3D置换操作）
            x_t = self.CBAM(x_t)
            
            # 修改1：使用卷积下采样
            for downsample in self.downsample_layers:
                x_t = F.relu(downsample(x_t))
            
            h_t[0], c_t[0], memory = self.cell_list[0](x_t, h_t[0], c_t[0], memory)
            
            for i in range(1, self.num_layers):
                h_prev_transformed = self.ffc_modules[i-1](h_t[i-1])
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_prev_transformed, h_t[i], c_t[i], memory
                )
            
            out = h_t[self.num_layers - 1]
            
            # 修改1：使用转置卷积上采样
            for upsample in self.upsample_layers:
                out = F.relu(upsample(out))
            
            # 修改2：使用1×1卷积输出
            x_gen = self.output_conv(out)
            next_frames.append(x_gen)
        
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        
        return next_frames
