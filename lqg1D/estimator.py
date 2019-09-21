import numpy as np
import random
import torch


def sampling_from_det_pol(env, n_samples, n_steps, det_pol):

    samples_list = []
    for i in range(0, n_samples):
        env.reset()
        for j in range(0, n_steps):
            state = env.get_state()
            det_pol.train()
            action = det_pol(torch.from_numpy(state).float())
            new_state, r, _, _ = env.step(action.detach().numpy())
            samples_list.append([state[0], action[0], r, new_state[0]])

    random.shuffle(samples_list)
    return samples_list


def get_mcrst_const(state, min, max, n_states):
    if state == min:
        return 0
    h = (max - min) / n_states
    index = 0
    while state > min + index * h:
        index = index + 1
    return index - 1


def get_mcrst_not_const(state, intervals):
    if state == intervals[-1][1]:
        return len(intervals) - 1
    index = 0
    for inter in intervals:
        if inter[0] <= state < inter[1]:
            return index
        else:
            index = index + 1


def estimate_mcrst_dist(samples_state, n_mcrst):
    accumulator = np.zeros(n_mcrst)
    for i in samples_state:
        accumulator[i] = accumulator[i] + 1

    # to avoid estimates equal to zero
    zeros = len(accumulator) - np.count_nonzero(accumulator)
    accumulator = map(lambda a: 1 if a == 0 else a, accumulator)
    return [a / (len(samples_state) + zeros) for a in accumulator]


def get_constant_intervals(min, max, n_mcrst):
    intervals = []
    h = (max - min) / n_mcrst
    while min < max:
        intervals.append([min, min + h])
        min += h
    return intervals


def update_parameter(param, learning_rate, grad):
    with torch.no_grad():
        update = param + learning_rate * grad
    if param.grad is not None:
        param.grad.zero_()
    return update



