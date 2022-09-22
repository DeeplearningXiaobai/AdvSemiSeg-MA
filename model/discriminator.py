import torch
import torch.nn as nn
from .spectral import SpectralNorm


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(FCDiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(num_classes, 64, 3, stride=1, padding=(1, 1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))
        self.classifier = SpectralNorm(nn.Conv2d(512, 1, 4, stride=2, padding=(1, 1)))
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(0.2)(self.conv1(m))
        m = nn.LeakyReLU(0.2)(self.conv2(m))
        m = nn.LeakyReLU(0.2)(self.conv3(m))
        m = nn.LeakyReLU(0.2)(self.conv4(m))
        m = nn.LeakyReLU(0.2)(self.conv5(m))
        m = nn.LeakyReLU(0.2)(self.conv6(m))
        m = nn.LeakyReLU(0.2)(self.conv7(m))
        m = self.classifier(m)
        m = self.upsample(m)
        return m


if __name__ == '__main__':
    x = torch.rand(1, 9, 224, 224)
    model = FCDiscriminator(num_classes=9)
    print(model)
    out = model(x)
    print(out.size())
