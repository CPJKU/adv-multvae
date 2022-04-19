from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from src.modules.polylinear import PolyLinear
from src.modules.multi_dae import MultiDAE
from src.modules.gradient_reversal import GradientReversalLayer


class MultiVAE(MultiDAE):
    def __init__(self, p_dims, q_dims=None, input_dropout_rate=0.5, latent_dropout_rate=0.,
                 normalize_inputs=True, **kwargs):
        """
        Variational Autoencoders for Collaborative Filtering - Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara
        https://arxiv.org/abs/1802.05814
        Attributes
        ---------
        p_dims  : list
            list of values that defines the structure of the network on the decoder side
        q_dims : list
            list of values that defines the structure of the network on the encoder side (Optional)
        input_dropout_rate: float
            dropout value
        """
        super().__init__(p_dims, q_dims, input_dropout_rate)
        self.latent_dropout = nn.Dropout(p=latent_dropout_rate)

        # Overwrite the encoder
        # variational auto encoder needs two times the hidden size, as the latent
        # space will be split up into a mean and a log(std) vector, with which we
        # sample from the multinomial distribution
        q_dims = self.q_dims.copy()
        q_dims[-1] *= 2

        self.normalize_inputs = normalize_inputs
        l1_weight_decay = kwargs.get("l1_weight_decay")
        self.encoder = PolyLinear(q_dims, nn.Tanh(), l1_weight_decay=l1_weight_decay)
        self.apply(self._init_weights)

    def encoder_forward(self, x):
        """
        Performs the encoding step of the variational auto-encoder
        :param x: the unnormalized data to encode
        :return: the sampled encoding + the KL divergence of the generated mean and std params
        """
        x = self.dropout(x)
        if self.normalize_inputs:
            x = F.normalize(x, 2, 1)
        x = self.encoder(x)
        mu, logvar = x[:, :self.latent], x[:, self.latent:]
        KL = self._calc_KL_div(mu, logvar)

        # Sampling #
        z = self._sampling(mu, logvar)
        return z, KL

    def forward(self, x):
        # Encoder #
        z, KL = self.encoder_forward(x)
        z = self.latent_dropout(z)

        # Decoder #
        z = self.decoder(z)

        return z, KL

    def _sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std * self.training
        return z

    @staticmethod
    def _calc_KL_div(mu, logvar):
        """
        Calculates the KL divergence of a multinomial distribution with the generated
        mean and std parameters
        """
        # Calculation for multinomial distribution with multivariate normal and standard normal distribution based on
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        # Mean is used as we may have different batch sizes
        return 0.5 * torch.mean(torch.sum(-logvar + torch.exp(logvar) + mu ** 2 - 1, dim=1))


class MultiVAEAdv(MultiVAE):
    """
    Adversarial network, based on
    "Adversarial Removal of Demographic Attributes from Text Data"
    https://www.aclweb.org/anthology/D18-1002/
    ========================
    Functionality we want to provide:
        - adjustable l, which controls the intensity of the reversal layer (GRL)
        - adjustable number of adversaries
        - adjustable depth of adversaries
        - (should we allow for different output shapes, e.g., s.t. we can try to predict different demographics?)
    """

    def __init__(self, p_dims, q_dims=None, input_dropout_rate=0.5, latent_dropout_rate=0., normalize_inputs=True,
                 use_adv_network=False, adv_config=None, **kwargs):
        """
        :param p_dims, q_dims, input_dropout, betacap, betasteps, beta_patience, latent_dropout_rat: see MultiVAE
        :param use_adv_network: whether to include a adversaries
        :param adv_config (dict): contains configurations for the adversaries:
            grad_scaling (float) ... impact of gradients on backpropagation
            adversaries (list) ... contains a list of dimensions for the linear dimensions.
                                   the latent dim is added automatically
        """
        super().__init__(p_dims, q_dims, input_dropout_rate, latent_dropout_rate, normalize_inputs, **kwargs)

        self.use_adv_network = use_adv_network
        self.adv_config = adv_config

        if use_adv_network:
            adv = OrderedDict()
            adv["grad_rev_layer"] = GradientReversalLayer(adv_config["grad_scaling"])
            adv["dropout"] = nn.Dropout(p=adv_config["latent_dropout"])
            self.adv_prep = nn.Sequential(adv)

            self.adversaries = nn.ModuleList()
            for adv in adv_config["adversaries"]:
                self.adversaries.append(PolyLinear([self.latent] + adv, nn.ReLU()))

    def reset_decoder(self):
        self.decoder.apply(self._init_weights)

    def forward(self, x):
        # Encoder #
        z, KL = self.encoder_forward(x)

        dz = self.latent_dropout(z)

        # Decoder #
        d = self.decoder(dz)

        ads = None
        # Adversaries #
        if self.use_adv_network:
            a = self.adv_prep(dz)
            ads = [adv(a) for adv in self.adversaries]

        return d, KL, ads  # returns (decoder output, KL loss, adversarial outputs - possible None)
