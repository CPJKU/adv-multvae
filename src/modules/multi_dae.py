import torch
from torch import nn
import torch.nn.functional as F

from src.modules.polylinear import PolyLinear


class MultiDAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, input_dropout=0.5, **kwargs):
        """
        Attributes
        ---------
        p_dims  : str
            list of values that defines the structure of the network on the decoder side
        q_dims : str
            list of values that defines the structure of the network on the encoder side (Optional)
        input_dropout: float
            dropout value
        """
        super().__init__()

        # Reading Parameters #
        self.p_dims = p_dims
        self.q_dims = q_dims if q_dims is not None else p_dims[::-1]
        assert self.q_dims[-1] == self.p_dims[0], "Latent dimension for p- and q-network mismatches."

        self.latent = self.p_dims[0]
        self.dropout = nn.Dropout(input_dropout)

        l1_weight_decay = kwargs.get("l1_weight_decay")
        self.encoder = PolyLinear(self.q_dims, nn.Tanh(), l1_weight_decay=l1_weight_decay)
        self.decoder = PolyLinear(self.p_dims, nn.Tanh(), l1_weight_decay=l1_weight_decay)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        gain = nn.init.calculate_gain('tanh')
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain)
            torch.nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x = F.normalize(x, 2, 1)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
