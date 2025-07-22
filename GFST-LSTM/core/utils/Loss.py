import random

import numpy as np
import torch



class BMSELoss(torch.nn.Module):

    def __init__(self):
        super(BMSELoss, self).__init__()
        self.w_l = [1, 2, 5, 10, 30]
        self.y_l = [0.283, 0.353, 0.424, 0.565, 1]

    def forward(self, x, y):
        w = y.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        return torch.mean(w * ((y - x) ** 2))


class BMAELoss(torch.nn.Module):

    def __init__(self):
        super(BMAELoss, self).__init__()
        self.w_l = [1, 2, 5, 10, 30]
        self.y_l = [0.283, 0.353, 0.424, 0.565, 1]

    def forward(self, x, y):
        w = y.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        return torch.mean(w * np.abs(y - x))


if __name__ == '__main__':
    data = BMSELoss()
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([0.4], requires_grad=True)
    loss = data(x, y)
    print(loss)

