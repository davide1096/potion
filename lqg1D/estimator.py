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

    # random.shuffle(samples_list)
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
    if len(intervals) > n_mcrst:
        del intervals[-1]
        intervals[n_mcrst - 1][1] = max
    return intervals


def update_parameter(param, learning_rate, grad):
    with torch.no_grad():
        update = param + learning_rate * grad
    if param.grad is not None:
        param.grad.zero_()
    return update


# working for non constant intervals!
def alternative_sampling(env, n_samples, n_steps, stoch_policy, intervals):
    samples_list = []
    for i in range(0, n_samples):
        env.reset()
        for j in range(0, n_steps):
            state = get_mcrst_not_const(env.get_state()[0], intervals)
            action = draw_action_weighted_policy(stoch_policy[state])
            new_state, r, _, _ = env.step(action)
            ns = get_mcrst_not_const(new_state[0], intervals)
            samples_list.append([state, action, r, ns])

    # random.shuffle(samples_list)
    return samples_list


def draw_action_weighted_policy(mcrst_policy):
    # rdm_number between 1 and the #samples started in that macrostate
    rdm_number = random.random() * sum(mcrst_policy.values())
    accumulator = 0
    for k in mcrst_policy.keys():
        accumulator += mcrst_policy[k]
        if accumulator >= rdm_number:
            # k is a key -> an action
            return float(k)

# ----------------------------------------------------------------------------------------------------


