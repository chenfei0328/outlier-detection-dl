import torch
import torch.nn.functional as F
from torch import nn


class AE(nn.Module):
    def __init__(self, in_dims=128, out_dims=128, latent_dims=3):
        super(AE, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.latent_dims = latent_dims

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dims, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )

        self.latent = nn.Linear(100, self.latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.out_dims),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x

    def loss_function(self, recon_x, x):
        MSE = nn.MSELoss()
        return MSE(recon_x, x)


class VAE(nn.Module):
    def __init__(self, in_dims=128, out_dims=128, latent_dims=3, eps=1e-4):
        super(VAE, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.latent_dims = latent_dims
        self.eps = eps

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dims, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(100, self.latent_dims)
        self.sigma_layer = nn.Linear(100, self.latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.out_dims),
            nn.Tanh(),
        )

    def encode_q_z(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        sigma = self.sigma_layer(h)
        sigma = F.softplus(sigma) + torch.FloatTensor([self.eps] * self.latent_dims)
        return mu, sigma

    def reparametrize(self, mu, sigma):
        std = sigma.mul(0.5).exp_()
        ksi = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            ksi = ksi.cuda()
        return ksi.mul(std).add_(mu)

    def decode_p_x(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, sigma = self.encode_q_z(x)
        z = self.reparametrize(mu, sigma)
        return self.decode_p_x(z), mu, sigma

    def loss_function(self, recon_x, x, mu, sigma):
        MSE = nn.MSELoss()
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(sigma.exp()).mul_(-1).add_(1).add_(sigma)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        #         print(MSE(recon_x, x))
        #         print(KLD)
        return MSE(recon_x, x) + KLD


# net = VAE()
# print(net)
# if torch.cuda.is_available():
#     net = net.cuda()
# print(net.loss_function(torch.Tensor([[0,0],[1,1]]),torch.Tensor([[1,2],[1,2]]), torch.Tensor([[1,2,1],[0.1,0.2,0.05]]), torch.Tensor([[1.2,1.5,2],[0.2,0.1,0]])))