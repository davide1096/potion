import torch.nn as nn
import torch


# log is a boolean that indicates if I need the grad or the grad_lo
def get_grad_tf_prob(w_xxdest, w_xxother, action, log):
    action = action.detach()
    w_xxdest = torch.tensor([w_xxdest], requires_grad=True)
    w_xxother = torch.tensor([x_oth for x_oth in w_xxother], requires_grad=True)
    den = torch.exp(w_xxdest * action) + torch.sum(torch.exp(w_xxother * action))
    prob = torch.exp(w_xxdest * action) / den
    out = torch.log(prob) if log else prob
    out.backward()
    return prob, w_xxdest.grad, w_xxother.grad
