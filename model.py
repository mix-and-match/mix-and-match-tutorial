import torch
from torch import nn
import random


class MixMatchVAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # the list containing the random indices sampled by "Sampling" operation
        self.sampled_indices = []
        self.complementary_indices = []

        # the encoder (note, it can be any neural network, e.g., GRU, Linear, ...)
        self.data_encoder = nn.Sequential(
            nn.Linear(args.input_dim, args.input_dim // 2),
            nn.BatchNorm1d(args.input_dim // 2),
            nn.ReLU(),
            nn.Linear(args.input_dim // 2, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
        )

        # the condition encoder (note, it can be any neural network, e.g., GRU, Linear, ...)
        self.condition_encoder = nn.Sequential(
            nn.Linear(args.condition_dim, args.condition_dim // 2),
            nn.BatchNorm1d(args.condition_dim // 2),
            nn.ReLU(),
            nn.Linear(args.condition_dim // 2, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
        )

        # layers to compute the data posterior
        self.mean = nn.Linear(args.hidden_dim, args.latent_dim)
        self.std = nn.Linear(args.hidden_dim, args.latent_dim)

        # layer to map the latent variable back to hidden size
        self.decode_latent = nn.Sequential(nn.Linear(args.latent_dim, args.hidden_dim), nn.ReLU())

        # the data decoder
        self.data_decoder = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.output_dim),
            nn.Sigmoid()
        )

    def encode_condition(self, condition):
        h = self.condition_encoder(condition)

        # sampled_indices are the ones that has been sampled by the "Sampling" operation
        # complementary_indices are the complementary set of indices that
        # has not been sampled in the "Sampling" operation.
        sampled_condition = h[:, self.sampled_indices]
        complementary_condition = h[:, self.complementary_indices]

        return sampled_condition, complementary_condition

    def encode_data(self, data):
        h = self.data_encoder(data)

        # sampled_indices are the ones that has been sampled by the "Sampling" operation
        # complementary_indices are the complementary set of indices that
        # has not been sampled in the "Sampling" operation.
        sampled_data = h[:, self.sampled_indices]
        complementary_data = h[:, self.complementary_indices]

        return sampled_data, complementary_data

    def encode(self, sampled_data, complementary_condition):

        # Resample
        # creating a new vector (the result of conditioning the encoder)
        # that the total size is args.hidden_dim
        fusion = torch.zeros(sampled_data.shape[0], sampled_data.shape[1] + complementary_condition.shape[1]).to(self.args.device)
        fusion[:, self.sampled_indices] = sampled_data
        fusion[:, self.complementary_indices] = complementary_condition

        # compute the mean and standard deviation of the approximate posterior
        mu = self.mean(fusion)
        sigma = self.std(fusion)

        # reparameterization for sampling from the approximate posterior
        return self.reparameterize(mu, sigma), mu, sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.args.device)
        return eps.mul(std).add_(mu)

    def decode(self, latent, sampled_condition):

        latent = self.decode_latent(latent)
        complementary_latent = latent[:, self.complementary_indices]

        # Resample
        fusion = torch.zeros(sampled_condition.shape[0], self.args.hidden_dim).to(self.args.device)
        fusion[:, self.sampled_indices] = sampled_condition
        fusion[:, self.complementary_indices] = complementary_latent

        # decode the data
        return self.data_decoder(fusion)

    def forward(self, data, condition, alpha):

        # The fist step is to perform "Sampling"
        # the parameter "alpha" is the pertubation rate. it is usually half of hidden_dim
        self.sampled_indices = list(random.sample(range(0, self.args.hidden_dim), alpha))
        self.complementary_indices = [i for i in range(self.args.hidden_dim) if i not in self.sampled_indices]

        # encode data and condition
        sampled_data, complementary_data = self.encode_data(data)
        sampled_condition, complementary_condition = self.encode_condition(condition)

        # VAE encoder
        z, mu, sigma = self.encode(sampled_data, complementary_condition)

        # VAE decoder
        decoded = self.decode(z, sampled_condition)

        return decoded, mu, sigma

    def sample(self, condition, alpha, z=None):
        # The fist step is to perform "Sampling"
        # the parameter "alpha" is the perturbation rate. it is usually half of hidden_dim
        self.sampled_indices = list(random.sample(range(0, self.args.hidden_dim), alpha))
        self.complementary_indices = [i for i in range(self.args.hidden_dim) if i not in self.sampled_indices]

        # encode the condition
        sampled_condition, complementary_condition = self.encode_condition(condition)

        # draw a sample from the prior distribution
        if z is None:
            z = torch.randn(condition.shape[0], self.args.latent_dim).normal_(0, 1).to(self.args.device)

        # VAE decoder
        generated = self.decode(z, sampled_condition)

        return generated
