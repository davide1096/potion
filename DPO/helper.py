import random
import numpy as np
import math


MAX_SAMPLES_IN_MCRST = 60

class Helper(object):

    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def big_mcrst_correction(self, cont):
        new_cont = {}
        for i in range(0, MAX_SAMPLES_IN_MCRST):
            rdm = random.randint(0, len(cont.keys()) - 1)
            index = list(cont.keys())[rdm]
            new_cont[index] = cont[index]
        return new_cont

    def getSeed(self):
        return self.seed


def get_mcrst(state, intervals, sink):

    if sink and state < intervals[0][0]:
        return -1
    elif sink and state > intervals[-1][1]:
        return len(intervals)

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


def calc_abs_reward_lqg(cont, action):
    rew = 0
    for act in cont.keys():
        rew += -0.5 * (cont[act]['state'] * cont[act]['state'] + action * action)
    return rew / len(cont.items())


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


def calc_abs_reward_cartpole(cont, action):
    return 1 if -0.2093 <= cont[action]['state'] <= 0.2093 else 0


def calc_abs_reward_minigolf(cont, action):
    action = np.clip(action, 0, 5)
    rew = 0
    for act in cont.keys():
        if action < np.sqrt(1.836 * cont[act]['state']):
            rew += -1
        elif action > np.sqrt(7.33 + 1.836 * cont[act]['state']):
            rew += -100
        else:
            rew += 0
    return rew / len(cont.items())


def estimate_J_from_samples(samples, gamma):
    acc = 0
    for sam in samples:
        g = 1
        for s in sam:
            # sum of discounted rewards of the initial samples.
            acc += g * s[2]
            g *= gamma
    return acc / len(samples)


def minigolf_reward_counter(samples):
    zeros = 0
    hundred = 0
    # max_action = 0
    failing_states = []
    for sam in samples:
        for s in sam:
            # if s[1] > max_action:
            #     max_action = s[1]
            if s[2] == 0:
                zeros += 1
            if s[2] == -100:
                hundred += 1
                failing_states.append(s[0])

    # print("Max action: {}".format(max_action))
    return zeros, hundred, failing_states


def interval_intersection(bounds):
    mins = [b[0] for b in bounds]
    maxs = [b[1] for b in bounds]
    if max(mins) <= min(maxs):
        return max(mins), min(maxs)
    else:
        # return min(mins), max(maxs)
        return None, None


def build_mcrst_from_samples(samples, n_mcrst, min_val, max_val):
    samples = flat_listoflists(samples)
    samples = sorted(samples)
    mcrst_size = math.floor(len(samples) / n_mcrst)
    INTERVALS = []

    while len(samples) >= 2 * mcrst_size:
        if len(INTERVALS) == 0:
            INTERVALS.append([min_val, samples[mcrst_size - 1][0]])
        else:
            INTERVALS.append([INTERVALS[-1][1], samples[mcrst_size - 1][0]])
        samples = samples[mcrst_size:]
    if len(samples) > 0:
        if len(INTERVALS) == 0:
            INTERVALS.append([min_val, max_val])
        else:
            INTERVALS.append([INTERVALS[-1][1], max_val])
    return INTERVALS


def get_constant_intervals(min_space, max_space, n_mcrst):
    dim = math.floor(((max_space - min_space) / n_mcrst) * 100)/100.0  # diameter of the macrostate
    remaining = round(((max_space - min_space) - dim * n_mcrst), 2)  # correction
    intervals = []
    counter = round(min_space + dim + math.floor(remaining/2 * 100)/100.0, 2)
    intervals.append([min_space, counter])
    while counter < round(max_space - 2 * dim, 2):
        intervals.append([counter, round(counter + dim, 2)])
        counter = round(counter + dim, 2)
    intervals.append([counter, max_space])
    return intervals


def get_constant_intervals(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN):
    mcrst = []
    for mins, maxs, n in zip(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN):  # for every dimension
        dim = math.floor(((maxs - mins) / n) * 100)/100.0  # diameter of the macrostate
        remaining = round(((maxs - mins) - dim * n), 2)  # correction
        intervals = []
        counter = round(mins + dim + math.floor(remaining/2 * 100)/100.0, 2)
        intervals.append([mins, counter])
        while counter <= round(maxs - 2 * dim, 2):
            intervals.append([counter, round(counter + dim, 2)])
            counter = round(counter + dim, 2)
        intervals.append([counter, maxs])
        mcrst.append(intervals)
    return mcrst