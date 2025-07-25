import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(SpatioTemporalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)

        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden * 7, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden * 4, height, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden * 3, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)  # x_t(4,16,16,16) -> (4,448,16,16)
        h_concat = self.conv_h(h_t)  # h_t(4,64,16,16) -> (4,256,16,16)
        m_concat = self.conv_m(m_t)  # m_t(4,64,16,16) -> (4,192,16,16)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)  # (4 64 16 16)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)  # (4 64 16 16)

        # 公式 # (4 64 16 16)
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)  # (4 128 16 16)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new