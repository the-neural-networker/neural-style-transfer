from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import random_split

from torchvision import transforms

import acoustics

from dataset import StyleTransferDataset


class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)

    def forward(self, x):
        conv1 = torch.relu(self.conv1(x))
        conv2 = torch.relu(self.conv2(conv1))
        out = F.max_pool2d(conv2, 2)
        return conv1, conv2, out


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)

    def forward(self, x):
        conv1 = torch.relu(self.conv1(x))
        conv2 = torch.relu(self.conv2(conv1))
        conv3 = torch.relu(self.conv3(conv2))
        conv4 = torch.relu(self.conv4(conv3))
        out = F.max_pool2d(conv4, 2)
        return conv1, conv2, conv3, conv4, out


class StyleTransferNet(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=64, input_shape=(1, 3, 512, 512)):
        super(StyleTransferNet, self).__init__()
        self.conv1 = ConvBlock1(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = ConvBlock1(in_channels=out_channels, out_channels=out_channels * 2)
        self.conv3 = ConvBlock2(in_channels=out_channels * 2, out_channels=out_channels * 4)
        self.conv4 = ConvBlock2(in_channels=out_channels * 4, out_channels=out_channels * 8)
        self.conv5 = ConvBlock2(in_channels=out_channels * 8, out_channels = out_channels * 8)
        # white noise image
        self.x = torch.tensor(acoustics.generator.white(input_shape[1] * input_shape[2] * input_shape[3]).reshape(input_shape), dtype=torch.float32, requires_grad=True)
        self.x = self.x.to(device="cuda")
        # print(self.x.shape, self.x.dtype)

    def forward(self, x):
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

    def training_step(self, batch, batch_idx):
        content, style = batch 
        content_feature_maps = self(content)
        style_feature_maps = self(style)
        x_feature_maps = self(self.x)

        # conv4_2 for content loss
        pl = content_feature_maps["conv4"][2]
        fl = x_feature_maps["conv4"][2]

        # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 for style loss
        a = []
        for i in range(1, 6):
            style_fl = style_feature_maps[f"conv{i}"][1]
            al = self.compute_gram_matrix(style_fl)
            a.append(al)

        g = []
        for i in range(1, 6):
            x_fl = x_feature_maps[f"conv{i}"][1]
            gl = self.compute_gram_matrix(x_fl)
            g.append(gl)

        loss = self.total_loss(fl, pl, g, a)
        self.log("train_loss", loss)
        return loss

    def content_loss(self, fl, pl):
        fl = fl.view(fl.shape[0], -1)
        pl = fl.view(pl.shape[0], -1)
        return 0.5 * ((fl - pl) ** 2).sum()

    def style_loss(self, g, a, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        e = 0 
        for gl, al, weight in zip(g, a, weights):
            constant = 1 / (4 * gl.shape[0] ** 2 * gl.shape[1] ** 2)
            el = weight * (constant * ((gl - al) ** 2).sum())
            e += el
        return e

    def total_loss(self, fl, pl, g, a, alpha=1, beta=10**3, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        return alpha * self.content_loss(fl, pl) + beta * self.style_loss(g, a, weights)

    def compute_gram_matrix(self, fl):
        fl = fl.view(fl.shape[0], -1)
        return torch.matmul(fl, fl.T)

    def get_result(self):
        return self.x

    def configure_optimizers(self):
        optimizer = optim.LBFGS(self.parameters(), lr=1)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = StyleTransferDataset()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=12)

    # ------------
    # model
    # ------------
    model = StyleTransferNet(in_channels=args.input_channels, out_channels=args.hidden_dim)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    result = model.get_result()
    print(result)


if __name__ == '__main__':
    cli_main()
