import torch
import math
import torch.nn as nn
import numpy as np
from lqg1D.estimator import update_parameter


class StochasticPolicy(nn.Module):

    def __init__(self, mu, omega, init_lr, min_action, max_action):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor([mu]))
        self.omega = nn.Parameter(torch.tensor([omega]))
        self.mu.requires_grad_(True)
        self.omega.requires_grad_(True)
        self.learning_rate = init_lr
        self.min_action = min_action
        self.max_action = max_action

    # def forward(self, action):
    #     sigma = torch.exp(self.omega)
    #     num = torch.exp(- ((action - self.mu) ** 2) / (2 * sigma ** 2))
    #     return num / (sigma * (2 * math.pi) ** 0.5)

    def update_parameters(self, grad_mu, grad_omega):
        new_mu = np.clip(update_parameter(self.mu, self.learning_rate, grad_mu), self.min_action, self.max_action)
        self.mu = nn.Parameter(torch.tensor([new_mu], requires_grad=True))
        self.omega = nn.Parameter(torch.tensor([update_parameter(self.omega, self.learning_rate, grad_omega)],
                                               requires_grad=True))

    def get_policy_prob(self, action):
        sigma = torch.exp(self.omega)
        num = torch.exp(- ((action - self.mu) ** 2) / (2 * sigma ** 2))
        return num / (sigma * (2 * math.pi) ** 0.5)

    def gradient_log_policy(self, action):
        out = torch.log(self.get_policy_prob(action))
        out.backward()
        return self.mu.grad, self.omega.grad

    # def get_parameters(self):
    #     return self.mu, self.omega

    def set_parameters(self, mu, omega):
        self.mu = nn.Parameter(torch.tensor([mu]))
        self.omega = nn.Parameter(torch.tensor([omega]))


class DeterministicPolicy(nn.Module):

    def __init__(self, initial_param):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([initial_param]))

    # def deterministic_action(self, state):
    #     return self.param * state

    def forward(self, state):
        return self.param * state
