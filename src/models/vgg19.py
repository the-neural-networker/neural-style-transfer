import torch
from torch import nn
import torch.nn.functional as F

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std

class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # conv1 = torch.relu(self.conv1(x))
        conv1 = self.relu1(self.conv1(x))
        # conv2 = torch.relu(self.conv2(conv1))
        conv2 = self.relu2(self.conv2(conv1))
        # out = F.max_pool2d(conv2, 2)
        out = self.max_pool2d(conv2)
        return conv1, conv2, out


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=False)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # conv1 = torch.relu(self.conv1(x))
        conv1 = self.relu1(self.conv1(x))
        # conv2 = torch.relu(self.conv2(conv1))
        conv2 = self.relu2(self.conv2(conv1))
        # conv3 = torch.relu(self.conv3(conv2))
        conv3 = self.relu3(self.conv3(conv2))
        # conv4 = torch.relu(self.conv4(conv3))
        conv4 = self.relu4(self.conv4(conv3))
        # out = F.max_pool2d(conv4, 2)
        out = self.max_pool2d(conv2)
        return conv1, conv2, conv3, conv4, out


class VGG19(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, mean=None, std=None):
        super(VGG19, self).__init__()
        self.norm = Normalization(mean=mean, std=std)
        self.conv1 = ConvBlock1(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = ConvBlock1(in_channels=out_channels, out_channels=out_channels * 2)
        self.conv3 = ConvBlock2(in_channels=out_channels * 2, out_channels=out_channels * 4)
        self.conv4 = ConvBlock2(in_channels=out_channels * 4, out_channels=out_channels * 8)
        self.conv5 = ConvBlock2(in_channels=out_channels * 8, out_channels = out_channels * 8)
            
    def forward(self, x):
        x = self.norm(x)
        conv1_1, conv1_2, out = self.conv1(x)
        conv2_1, conv2_2, out = self.conv2(out)
        conv3_1, conv3_2, conv3_3, conv3_4, out = self.conv3(out)
        conv4_1, conv4_2, conv4_3, conv4_4, out = self.conv4(out)
        conv5_1, conv5_2, conv5_3, conv5_4, out = self.conv5(out)
        outputs = {
            "conv1": [conv1_1, conv1_2],
            "conv2": [conv2_1, conv2_2],
            "conv3": [conv3_1, conv3_2, conv3_3, conv3_4],
            "conv4": [conv4_1, conv4_2, conv4_3, conv4_4],
            "conv5": [conv5_1, conv5_2, conv5_3, conv5_4],
            "out" : out
        }

        return outputs 