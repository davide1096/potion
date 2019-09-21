import torch
import numpy as np


def get_tf_prob(w, b, arriving_mcrst, action):
    w = w.detach().numpy()
    b = b.detach().numpy()
    den = np.sum(np.exp(w * action + b))
    return np.exp(w[arriving_mcrst] * action + b[0][arriving_mcrst]) / den


# log is a boolean that indicates if I need the grad or the grad_log
def get_grad_tf_prob(w, b, arriving_mcrst, action, log):
    action = action.detach()
    den = torch.sum(torch.exp(w * action + b))
    prob = torch.exp(w[arriving_mcrst] * action + b[0][arriving_mcrst]) / den
    out = torch.log(prob) if log else prob
    out.backward()
    return prob
