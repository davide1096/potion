import random

SEED = 42
random.seed(SEED)

MAX_SAMPLES_IN_MCRST = 20


def big_mcrst_correction(cont):
    new_cont = {}
    for i in range(0, MAX_SAMPLES_IN_MCRST):
        rdm = random.randint(0, len(cont.keys()) - 1)
        index = list(cont.keys())[rdm]
        new_cont[index] = cont[index]
    return new_cont


def get_mcrst(state, intervals):
    if state >= intervals[-1][1]:   # above the upper bound of the state space
        return len(intervals) - 1
    if state <= intervals[0][0]:    # below the lower bound of the state space
        return 0
    index = 0                       # inside the state space
    for inter in intervals:
        if inter[0] <= state < inter[1]:
            return index
        else:
            index = index + 1


def calc_abs_reward(intervals, mcrst, action):
    mcrst_bounds = intervals[mcrst]
    mcrst_mean = (mcrst_bounds[0] + mcrst_bounds[1])/2
    return -0.5 * (mcrst_mean * mcrst_mean + action * action)


def count_actions(container):
    n_actions = 0
    for i in range(0, len(container)):
        n_actions += len(list(container[i].keys()))
    return n_actions


def normalize_array(array):
    den = sum(array)
    return [p / den for p in array]


def flat_listoflists(list):
    return [item for sublist in list for item in sublist]


def estimate_abstractJ(fict_samples, gamma, intervals):
    acc = 0
    for sample in fict_samples:
        g = 1
        for s in sample:
            abs_rew = calc_abs_reward(intervals, get_mcrst(s[0], intervals), s[1])
            acc += g * abs_rew
            g *= gamma
    return acc / len(fict_samples)
