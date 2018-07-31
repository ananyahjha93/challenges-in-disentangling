import torch
import torch.nn as nn
from collections import OrderedDict

from itertools import cycle
from torchvision import datasets
from utils import transform_config
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, nv_dim, nc_dim):
        super(Encoder, self).__init__()

        self.conv_model = nn.Sequential(OrderedDict([
            ('convolution_1',
             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_1_in', nn.InstanceNorm2d(num_features=16, track_running_stats=True)),
            ('ReLU_1', nn.ReLU(inplace=True)),

            ('convolution_2',
             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_2_in', nn.InstanceNorm2d(num_features=32, track_running_stats=True)),
            ('ReLU_2', nn.ReLU(inplace=True)),

            ('convolution_3',
             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_3_in', nn.InstanceNorm2d(num_features=64, track_running_stats=True)),
            ('ReLU_3', nn.ReLU(inplace=True))
        ]))

        # Nv
        self.fully_connected_varying_factor = nn.Linear(in_features=256, out_features=nv_dim, bias=True)

        # Nc
        self.fully_connected_common_factor = nn.Linear(in_features=256, out_features=nc_dim, bias=True)

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        nv_latent_space = self.fully_connected_varying_factor(x)
        nc_latent_space = self.fully_connected_common_factor(x)

        return nv_latent_space, nc_latent_space


class Decoder(nn.Module):
    def __init__(self, nv_dim, nc_dim):
        super(Decoder, self).__init__()

        # Nv
        self.fully_connected_varying_factor = nn.Linear(in_features=nv_dim, out_features=256, bias=True)

        # Nc
        self.fully_connected_common_factor = nn.Linear(in_features=nc_dim, out_features=256, bias=True)

        self.deconv_model = nn.Sequential(OrderedDict([
            ('deconvolution_1',
             nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True)),
            ('deconvolution_1_in', nn.InstanceNorm2d(num_features=32, track_running_stats=True)),
            ('ReLU_1', nn.ReLU(inplace=True)),

            ('deconvolution_2',
             nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True)),
            ('deconvolution_2_in', nn.InstanceNorm2d(num_features=16, track_running_stats=True)),
            ('ReLU_2', nn.ReLU(inplace=True)),

            ('deconvolution_3',
             nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True)),
            ('sigmoid_final', nn.Sigmoid())
        ]))

    def forward(self, nv_latent_space, nc_latent_space):
        nv_latent_space = self.fully_connected_varying_factor(nv_latent_space)
        nc_latent_space = self.fully_connected_common_factor(nc_latent_space)

        x = torch.cat((nv_latent_space, nc_latent_space), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self.deconv_model(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_model = nn.Sequential(OrderedDict([
            ('convolution_1',
             nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_1_in', nn.InstanceNorm2d(num_features=32)),
            ('LeakyReLU_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('convolution_2',
             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_2_in', nn.InstanceNorm2d(num_features=64)),
            ('LeakyReLU_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('convolution_3',
             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_3_in', nn.InstanceNorm2d(num_features=128)),
            ('LeakyReLU_3', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        ]))

        self.fully_connected_model = nn.Sequential(OrderedDict([
            ('output', nn.Linear(in_features=512, out_features=2, bias=True))
        ]))

    def forward(self, image_1, image_2):
        x = torch.cat((image_1, image_2), dim=1)
        x = self.conv_model(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fully_connected_model(x)

        return x


class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=z_dim, out_features=256, bias=True)),
            ('fc_1_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_2', nn.Linear(in_features=256, out_features=256, bias=True)),
            ('fc_2_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_3', nn.Linear(in_features=256, out_features=num_classes, bias=True))
        ]))

    def forward(self, z):
        x = self.fc_model(z)

        return x


if __name__ == '__main__':
    encoder = Encoder(64, 64)
    decoder = Decoder(64, 64)
    discriminator = Discriminator()

    mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(mnist, batch_size=16, shuffle=True, num_workers=0, drop_last=True))

    image_batch, labels_batch = next(loader)

    nv_latent_space, nc_latent_space = encoder(Variable(image_batch))
    reconstructed_image = decoder(nv_latent_space, nc_latent_space)

    adversarial_output = discriminator(reconstructed_image, reconstructed_image)
    print(adversarial_output.size())
