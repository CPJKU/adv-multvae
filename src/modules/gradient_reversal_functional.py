from torch.autograd import Function


class GRL_(Function):
    """
    Gradient reversal functional
    Unsupervised Domain Adaptation by Backpropagation - Yaroslav Ganin, Victor Lempitsky
    https://arxiv.org/abs/1409.7495
    """

    @staticmethod
    def forward(ctx, input, grad_scaling):
        ctx.grad_scaling = grad_scaling
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # need to return a gradient for each input parameter of the forward() function
        # for parameters that don't require a gradient, we have to return None
        # see https://stackoverflow.com/a/59053469
        return -ctx.grad_scaling * grad_output, None


grl = GRL_.apply
