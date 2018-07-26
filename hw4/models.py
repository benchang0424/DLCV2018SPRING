import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),   # 32*32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.005, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 16*16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.005, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 8*8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.005, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 4*4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.005, inplace=True)
        )
        self.z_mean = nn.Linear(2048, self.latent_dim)
        self.z_logvar = nn.Linear(2048, self.latent_dim)
        self.z_decode = nn.Linear(self.latent_dim, 2048)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True), 
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(3)
        )
        self.tanh = nn.Tanh()
    def encode(self, x):
        output = self.encoder(x).view(-1, 2048)
        return self.z_mean(output), self.z_logvar(output)

    def decode(self, z):
        fc_out = self.z_decode(z).view(-1, 128, 4, 4)
        de_out = self.decoder(fc_out)
        return self.tanh(de_out)/2.0+0.5

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
                     
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# for GAN
class Generator(nn.Module):
    def __init__(self, ngf=64):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
            # state size. (ngf) x 32 x 32
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, x):
        hidden = self.decoder(x)
        output = self.output(hidden)/2.0+0.5
        return output
    

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 4 x 4
        )
        self.output = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        hidden = self.main(x)
        output = self.output(hidden)

        return output.view(-1, 1).squeeze(1)


# for ACGAN
class ACGenerator(nn.Module):
    def __init__(self, ngf=64):
        super(ACGenerator, self).__init__()
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100+1, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
            # state size. (ngf) x 32 x 32
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, x):
        hidden = self.decoder(x)
        output = self.output(hidden)/2.0+0.5
        return output


class ACDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super(ACDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 4 x 4
        )
        
        self.out_dis = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.out_aux = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        hidden = self.main(x)
        output_dis = self.out_dis(hidden)
        output_aux = self.out_aux(hidden)
        return output_dis, output_aux




