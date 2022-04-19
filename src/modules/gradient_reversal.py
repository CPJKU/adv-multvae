from torch.nn import Module
from src.modules.gradient_reversal_functional import grl


class GradientReversalLayer(Module):
    def __init__(self, grad_scaling):
        """
        Gradient reversal layer
        Unsupervised Domain Adaptation by Backpropagation - Yaroslav Ganin, Victor Lempitsky
        https://arxiv.org/abs/1409.7495

        :param grad_scaling: the scaling factor that should be applied on the gradient in the backpropagation phase
        """
        super().__init__()
        self.grad_scaling = grad_scaling

    def forward(self, input):
        return grl(input, self.grad_scaling)

    def extra_repr(self) -> str:
        return f"grad_scaling={self.grad_scaling}"
