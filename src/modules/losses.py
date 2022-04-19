import numpy as np
import torch
import torch.nn.functional as F


class VAE_loss:
    def __init__(self, beta=None, beta_cap=0.5, beta_steps=2000, beta_patience=5):
        """
        :param beta: if provided, the beta value will be kept at this value
        :param beta_cap: maximum value beta can reach
        :param beta_steps: maximum number of beta annealing steps
        :param beta_patience: number of steps with no improvement after which beta annealing should be halted
        """
        super().__init__()

        self.beta = beta
        self.beta_cap = beta_cap
        self.beta_steps = beta_steps
        self._curr_beta = 0

        if beta is not None:
            self._curr_beta = beta

        # Parameters for beta annealing
        self.patience = beta_patience
        self._n_steps_wo_increase = 0
        self._best_score = -np.inf

    def __call__(self, logits, KL, y):
        prob = F.log_softmax(logits, dim=1)

        neg_ll = - torch.mean(torch.sum(prob * y, dim=1))
        weighted_KL = self._curr_beta * KL
        loss = neg_ll + weighted_KL

        return loss, neg_ll, weighted_KL

    def beta_step(self, score):
        """
        Performs the annealing procedure for the beta parameter
        Described in "Variational Autoencoders for Collaborative Filtering", Section 2.2.2
        :param score: The score used to determine whether to keep increasing the beta parameter
        :return: The current beta parameter, either updated or still from the previous call
        """
        if self.beta is not None:
            return self._curr_beta

        if self._n_steps_wo_increase > self.patience:
            return self._curr_beta

        # Even if validation score does not improve, we will still increase beta
        if self._best_score > score:
            self._n_steps_wo_increase += 1
        else:
            self._best_score = score
            self._n_steps_wo_increase = 0

        self._curr_beta += self.beta_cap / self.beta_steps
        self._curr_beta = min(self.beta_cap, self._curr_beta)
        return self._curr_beta
