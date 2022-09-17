'''
检测器模型-- 将真实图片标记为1， 将虚假图片标记为0
'''

import torch.nn as nn


class D_model(nn.Module):
    def __init__(self):
        super(D_model, self).__init__()
        self.main = nn.Sequential(
            # input： 3 x 64 x 64, output: 64 x 32 x 32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # negative_slope-控制负斜率的角度

            # input: 64 x 32 x 32, output: 128 x 16 x 16
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # input: 128 x 16 x 16, output : 256 x 8 x 8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # input : 256 x 8 x 8, output : 512 x 4 x 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            #  展平：8192
            nn.Flatten(),
            # 变成 1
            nn.Linear(8192, 1),
            # sigmoid
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)