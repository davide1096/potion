import numpy as np


def get_mcrst(state, min, max, n_states):
    if state == min:
        return 0
    h = (max - min) / n_states
    index = 0
    while state > min + index * h:
        index = index + 1
    return index - 1


def estimate_mcrst_dist(samples, n_macrostates, regular, min_state, max_state, intervals=None):
    # regular is a boolean telling if all the macrostates have the same dimension
    accumulator = np.zeros(n_macrostates)
    if regular:
        index = [get_mcrst(sample, min_state, max_state, n_macrostates) for sample in samples]
        for i in index:
            accumulator[i] = accumulator[i] + 1

        # to avoid estimates equal to zero
        zeros = len(accumulator) - np.count_nonzero(accumulator)
        accumulator = map(lambda a: 1 if a == 0 else a, accumulator)
        return [a / (len(samples) + zeros) for a in accumulator]
    else:
        pass
        # TODO
