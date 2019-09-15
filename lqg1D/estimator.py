import numpy as np


def get_mcrst_const(state, min, max, n_states):
    if state == min:
        return 0
    h = (max - min) / n_states
    index = 0
    while state > min + index * h:
        index = index + 1
    return index - 1

def get_mcrst_not_const(state, intervals):
    index = 0
    for int in intervals:
        if int[0] <= state < int[1]:
            return index
        else:
            index = index + 1


def estimate_mcrst_dist(samples_state, n_macrostates, constant_intervals, min_state, max_state, intervals=None):
    # regular is a boolean telling if all the macrostates have the same dimension
    accumulator = np.zeros(n_macrostates)
    if constant_intervals:
        index = [get_mcrst_const(sample, min_state, max_state, n_macrostates) for sample in samples_state]
    else:
        index = [get_mcrst_not_const(sample, intervals) for sample in samples_state]

    for i in index:
        accumulator[i] = accumulator[i] + 1

    # to avoid estimates equal to zero
    zeros = len(accumulator) - np.count_nonzero(accumulator)
    accumulator = map(lambda a: 1 if a == 0 else a, accumulator)
    return [a / (len(samples_state) + zeros) for a in accumulator]
    # TODO



