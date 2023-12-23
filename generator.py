import torch
from torch import nn
import config
import random
# from matplotlib.pyplot import imshow
from timeit import default_timer as timer



def print_train_time(start: float, end: float, device: torch.device):
    """
    Print difference between start and end time
    :param start:
    :param end:
    :param device:
    :return: float: time between start and end in seconds (higher is longer)
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# class CGANBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=2, down=True, use_dropout=False):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1,
#                       bias=False)
#             if down else
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),  # try both and choose the best option of normalization
#             # nn.InstanceNorm2d(out_channels, affine=True),
#             nn.LeakyReLU(0.2) if down else nn.ReLU())
#         self.useDropout = use_dropout
#
#     def forward(self, x):
#         x = self.block(x)
#         return nn.Dropout(0.5)(x) if self.useDropout else x


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, dropout=False):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.block(x)
        return torch.cat([x, skip_input], dim=1)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDownBlock(in_channels, 64, norm=False)
        self.down2 = UNetDownBlock(64, 128)
        self.down3 = UNetDownBlock(128, 256)
        self.down4 = UNetDownBlock(256, 512)
        self.down5 = UNetDownBlock(512, 512)
        self.down6 = UNetDownBlock(512, 512)
        self.down7 = UNetDownBlock(512, 512)
        self.down8 = UNetDownBlock(512, 512, norm=False, dropout=True)
        self.up1 = UNetUpBlock(512, 512, dropout=True)
        self.up2 = UNetUpBlock(1024, 512, dropout=True)
        self.up3 = UNetUpBlock(1024, 512, dropout=True)
        self.up4 = UNetUpBlock(1024, 512, dropout=True)
        self.up5 = UNetUpBlock(1024, 256)
        self.up6 = UNetUpBlock(512, 128)
        self.up7 = UNetUpBlock(256, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


# class Generator1(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super().__init__()
#         self.initialInputBlock = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode='reflect', bias=False),
#             nn.LeakyReLU(0.2))
#         self.downBlock1 = CGANBlock(64, 128)
#         self.downBlock2 = CGANBlock(128, 256)
#         self.downBlock3 = CGANBlock(256, 512)
#         self.downBlock4 = CGANBlock(512, 512)
#         self.downBlock5 = CGANBlock(512, 512)
#         self.downBlock6 = CGANBlock(512, 512)
#         self.bottleNeckBlock = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1, padding_mode='reflect'),
#                                              nn.ReLU())
#         self.upBlock1 = CGANBlock(512, 512, down=False, use_dropout=True)
#         self.upBlock2 = CGANBlock(1024, 512, down=False, use_dropout=True)
#         self.upBlock3 = CGANBlock(1024, 512, down=False, use_dropout=True)
#         self.upBlock4 = CGANBlock(1024, 512, down=False)
#         self.upBlock5 = CGANBlock(1024, 256, down=False)
#         self.upBlock6 = CGANBlock(512, 128, down=False)
#         self.upBlock7 = CGANBlock(256, 64, down=False)
#         self.finalBlock = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 1))
#
#     def forward(self, x):
#         down1 = self.initialInputBlock(x)
#         down2 = self.downBlock1(down1)
#         down3 = self.downBlock2(down2)
#         down4 = self.downBlock3(down3)
#         down5 = self.downBlock4(down4)
#         down6 = self.downBlock5(down5)
#         down7 = self.downBlock6(down6)
#         bottle_neck = self.bottleNeckBlock(down7)
#         up1 = self.upBlock1(bottle_neck)
#         up2 = self.upBlock2(torch.cat([up1, down7], dim=1))
#         up3 = self.upBlock3(torch.cat([up2, down6], dim=1))
#         up4 = self.upBlock4(torch.cat([up3, down5], dim=1))
#         up5 = self.upBlock5(torch.cat([up4, down4], dim=1))
#         up6 = self.upBlock6(torch.cat([up5, down3], dim=1))
#         up7 = self.upBlock7(torch.cat([up6, down2], dim=1))
#         return self.finalBlock(torch.cat([up7, down1], dim=1))

def main():
  gen = Generator().to(config.DEVICE)
  # gen.train()
  # torch.random.manual_seed(42)
  t = torch.randn(size=(1, 3, 256, 256))
  # plt.imshow(tensor.squeeze(0).permute(1, 2, 0))
  # plt.show()
  # plt.imshow(tensor.permute(1, 2, 0))
  # plt.show()
  # print(tensor.shape)
  # gen.eval()
  # with torch.inference_mode():
  #     pass

  gen.eval()
  train_time_start_on_cpu = timer()
  with torch.no_grad():
      result = gen(t.to(config.DEVICE))
  train_time_end_on_cpu = timer()
  from torchvision.utils import save_image

  print(result.shape)
  result = result[0]
  print(result.shape)
  # save_image(result, "./prod/result.png")
  # plt.imshow(result.squeeze(0).detach().permute(1, 2, 0).numpy())
  # plt.show()
  print_train_time(train_time_start_on_cpu, train_time_end_on_cpu, config.DEVICE)
  # print(gen)
  # print(list(gen.parameters()))
  # print(gen.state_dict())

  # try_tensor = torch.randn(size=(3, 256, 256))
  # out = gen(try_tensor.unsqueeze(0))
  # print(f"----------------------------------------------------------------------------\n{out}")

  # torch.manual_seed(42)
  # images = torch.randn(size=(32, 3, 64, 64))
  # test_image = images[0]
  # print(f"Image batch shape: {images.shape}")
  # print(f"Single image shape: {test_image.shape}")
  # print(f"Test image:\n {test_image}")
  # # Create a single conv2d layer
  # torch.manual_seed(42)
  # conv_layer = nn.Conv2d(in_channels=3,
  #                        out_channels=10,
  #                        kernel_size=3,
  #                        padding=0)
  # conv_output = conv_layer(test_image)
  # print('---------------------------------------------------------------')
  # print(conv_output)
  # print(torch.__version__)


if __name__ == "__main__":
    main()