'''
生成器模型--造假
'''
import torch.nn as nn


class G_model(nn.Module):
    def __init__(self):
        super(G_model, self).__init__()
        self.main = nn.Sequential(
            # input: 100 x 1 x 1， output: 512 x 4 x 4, （n-1)*s - 2p + k
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # input: 512 x 4 x4, output : 256 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # input: 256 x 8 x8, output: 128 x 16 x 16
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            # input : 128 x 16 x 16, output: 64 x 32 x 32
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            # 转置卷积，上采样 input: 64 x 32 x 32, output: 3 x 64 x 64
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            # tanh
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)