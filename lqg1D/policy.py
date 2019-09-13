import torch
import math
import torch.nn as nn


class StochasticPolicy(nn.Module):

    def __init__(self, mu, omega, init_lr):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor([mu]))
        self.omega = nn.Parameter(torch.tensor([omega]))
        self.mu.requires_grad_(True)
        self.omega.requires_grad_(True)
        self.learning_rate = init_lr

    # def forward(self, action):
    #     sigma = torch.exp(self.omega)
    #     num = torch.exp(- ((action - self.mu) ** 2) / (2 * sigma ** 2))
    #     return num / (sigma * (2 * math.pi) ** 0.5)

    def update_parameters(self, grad_mu, grad_omega):
        if self.mu.grad is not None:
            self.mu.grad.zero_()
        if self.omega.grad is not None:
            self.omega.grad.zero_()
        with torch.no_grad():
            self.mu += self.learning_rate * grad_mu
            self.omega += self.learning_rate * grad_omega

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


class DeterministicPolicy(nn.Module):

    def __init__(self, initial_param):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([initial_param]))

    # def deterministic_action(self, state):
    #     return self.param * state

    def forward(self, state):
        return self.param * state
