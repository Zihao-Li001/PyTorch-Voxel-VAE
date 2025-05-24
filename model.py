import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from operator import __add__

# TODO: Remove
class Conv3dSamePadding(nn.Conv3d):
    def __init__(self,*args,**kwargs):
        super(Conv3dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_3d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_3d(input), self.weight, self.bias)

# input_shape = (1, 32, 32, 32)
# z_dim = 128

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        in_channels = 1
        self.enc_conv1 = nn.Sequential(
          nn.Conv3d(in_channels, 8, kernel_size=3, stride=1, padding=0),
          nn.BatchNorm3d(8),
          nn.LeakyReLU()
        )

        in_channels = 8
        self.enc_conv2 = nn.Sequential(
        #   nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=0),
          Conv3dSamePadding(in_channels, 16, kernel_size=3, stride=2), # padding=same in keras
          nn.BatchNorm3d(16),
          nn.LeakyReLU()
        )

        in_channels = 16
        self.enc_conv3 = nn.Sequential(
          nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=0),
          nn.BatchNorm3d(32),
          nn.LeakyReLU()
        )

        in_channels = 32
        self.enc_conv4 = nn.Sequential(
        #   nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=0),
          Conv3dSamePadding(in_channels, 64, kernel_size=3, stride=2), # padding=same in keras
          nn.BatchNorm3d(64),
          nn.LeakyReLU()
        )

        self.enc_fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21952, 343), # Calculate 21952
            nn.BatchNorm1d(343),
            nn.LeakyReLU()
        )

        # self.mu = nn.Sequential(
        #     nn.Linear(343, 128),
        #     nn.BatchNorm1d(128)
        # )

        # self.sigma = nn.Sequential(
        #     nn.Linear(343, 128),
        #     nn.BatchNorm1d(128)
        # )
        self.mu = nn.Linear(343, 128) # 移除batchNorm
        self.sigma = nn.Linear(343, 128)


        # Mul mu and sigma
        # z = Lambda(
        #     sampling,
        #     output_shape = (z_dim, ))([mu, sigma])
        # encoder = Model(enc_in, [mu, sigma, z])

        # decoder
        self.dec_fc1 = nn.Sequential(
            nn.Linear(128, 343),
            nn.BatchNorm1d(343),
            nn.LeakyReLU()
        )

        self.dec_unflatten = nn.Sequential(
            nn.Unflatten(1, (1, 7, 7, 7))
        )

        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )

        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU()
        )

        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm3d(8),
            nn.LeakyReLU()
        )

        self.dec_conv5 = nn.Sequential(
            nn.ConvTranspose3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU()
        )

        self.decoder_output = nn.Sequential(
            nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU()
            nn.LeakyReLU() # 保证输出在0-1之间
        )

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(mu)
        return mu + torch.exp(0.5 * sigma) * epsilon

    def encode(self, x):
        encoder = self.enc_conv1(x)
        encoder = self.enc_conv2(encoder)
        encoder = self.enc_conv3(encoder)
        encoder = self.enc_conv4(encoder)

        fc1 = self.enc_fc1(encoder)

        mu = self.mu(fc1)
        sigma = self.sigma(fc1)

        z = self.reparameterize(mu, sigma)

        return mu, sigma, z

    def decode(self, x):
        decoder = self.dec_fc1(x)
        
        decoder = self.dec_unflatten(decoder)

        decoder = self.dec_conv1(decoder)
        decoder = self.dec_conv2(decoder)
        decoder = self.dec_conv3(decoder)
        decoder = self.dec_conv4(decoder)
        decoder = self.dec_conv5(decoder)

        return decoder

    def forward(self, x):
        # print(f"Input shape: {x.shape}") # Debug @ Check input shape, sh
        mu, sigma, z = self.encode(x)

        dec_conv5 = self.decode(z)

        return self.decoder_output(dec_conv5), mu, sigma

    def loss(self, inputs, outputs, mu, logvar, beta):
        outputs_sigmoid = torch.sigmoid(outputs)

        outputs_clip = torch.clamp(outputs, -10, 10)
        # weights = torch.where(inputs > 0.5, 0.98, 0.02)
        # bce = -(weights * inputs * torch.log(outputs_clip) + (1.0 - inputs) * torch.log(1.0 - outputs_clip))
        # bce = bce.mean()
        recon_loss = F.binary_cross_entropy(outputs, inputs, weight=torch.tensor(49.0).to(inputs.device))
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.size(0) # KLD loss
        total_loss = recon_loss + beta * kld_loss

        return total_loss, recon_loss, kld_loss
    

    # def loss(self, inputs, outputs, mu, sigma, beta):

    #     outputs_clip = torch.clip(torch.sigmoid(outputs), 1e-7, 1.0 - 1e-7)
    #     bce = -(98.0 * inputs * torch.log(outputs_clip) + 2.0 * (1.0 - inputs) * torch.log(1.0 - outputs_clip)) / 100.0
    #     bce = bce.mean()

    #     # KLD
    #     kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / inputs.size(0) 

    #     return bce + beta*kld , bce, kld