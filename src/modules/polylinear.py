from torch import nn
from collections import OrderedDict


class PolyLinear(nn.Module):
    def __init__(self, layer_config: list, activation_fn, output_fn=None, input_dropout=None, l1_weight_decay=None):
        """
        Helper module to easily create multiple linear layers and pass an
        activation through them
        :param layer_config: A list containing the in_features and out_features for the linear layers
                             Example: [100,50,2] would create two linear layers: Linear(100, 50) and Linear(50, 2),
                             whereas the output of the first layer is used as input for the second layer
        :param activation_fn: The activation function to use between layers
        :param output_fn: (optional) The function to apply on the output, e.g. softmax
        :param input_dropout: A possible dropout to apply to the input before passing it through the layers
        :param l1_weight_decay: Additional L1 weight normalization to induce sparsity in the layers
        """
        super().__init__()

        assert len(layer_config) > 1, "For a linear network, we at least need one " \
                                      "input and one output dimension"

        self.layer_config = layer_config
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        self.n_layers = len(layer_config) - 1

        layer_dict = OrderedDict()

        if input_dropout is not None:
            layer_dict["input_dropout"] = nn.Dropout(p=input_dropout)

        for i, (d1, d2) in enumerate(zip(layer_config[:-1], layer_config[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            if l1_weight_decay and l1_weight_decay > 0.0:
                from torchlayers.regularization import L1
                layer = L1(layer, weight_decay=l1_weight_decay)

            layer_dict[f"linear_{i}"] = layer
            if i < self.n_layers - 1:
                # only add activation functions in intermediate layers
                layer_dict[f"{activation_fn.__class__.__name__.lower()}_{i}"] = activation_fn

        if self.output_fn is not None:
            layer_dict[f"{output_fn.__class__.__name__.lower()}"] = self.output_fn

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        x = self.layers(x)
        return x
