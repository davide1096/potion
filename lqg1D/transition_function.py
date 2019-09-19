import torch
import numpy as np

# log is a boolean that indicates if I need the grad or the grad_log
# def get_grad_tf_prob(w_xxdest, w_xxother, action, log):
#     action = action.detach()
#     w_xxdest = torch.tensor([w_xxdest], requires_grad=True)
#     w_xxother = torch.tensor([x_oth for x_oth in w_xxother], requires_grad=True)
#     den = torch.exp(w_xxdest * action) + torch.sum(torch.exp(w_xxother * action))
#     prob = torch.exp(w_xxdest * action) / den
#     out = torch.log(prob) if log else prob
#     out.backward()
#     return prob, w_xxdest.grad, w_xxother.grad


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
