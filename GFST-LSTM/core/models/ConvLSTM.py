import math
import torch
import torch.nn as nn
from core.layers.ConvLSTMCell import ConvLSTMCell


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel  # 1*1*1
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size  # 16
        height = configs.img_height // configs.patch_size

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                        configs.stride)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(self.frame_channel, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)

    def forward(self, frames, mask_true):
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        height = frames.shape[3]  # 64/4=16
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
        x_gen = None
        # h_t、c_t、memory初始化为0
        for i in range(self.num_layers):
            zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)  # （4，64，16，16）
            h_t.append(zeros)
            c_t.append(zeros)
        # memory = torch.zeros([batch_size, self.num_hidden[0], height, width]).to(self.configs.device)  # （4，64，16，16）

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]  # (4,1,64,64)
            else:
                time_diff = t - self.configs.input_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            frames_feature = net
            x_t = self.merge(frames_feature)

            h_t[0], c_t[0] = self.cell_list[0](x_t, h_t[0], c_t[0])

            # 得到当前时刻的H_t、C_t、M
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            out = h_t[self.num_layers - 1]

            x_gen = self.srcnn(out)
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
