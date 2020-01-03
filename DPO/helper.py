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

    mcrst = []
    for dim in range(len(intervals)):
        dim_int = intervals[dim]

        if sink and state[dim] < dim_int[0][0]:
            mcrst.append(-1)
        elif sink and state[dim] > dim_int[-1][1]:
            mcrst.append(len(dim_int))

        elif state[dim] >= dim_int[-1][1]:   # above the upper bound of the state space
            mcrst.append(len(dim_int) - 1)
        elif state[dim] <= dim_int[0][0]:    # below the lower bound of the state space
            mcrst.append(0)

        else:
            index = 0                       # inside the state space
            for inter in dim_int:
                if inter[0] <= state[dim] < inter[1]:
                    mcrst.append(index)
                    break
                else:
                    index = index + 1
    return mcrst


# --> helper function to compute the index in container related to the exact macrostate <---

def get_multidim_mcrst(multi_mcrst, intervals):
    mcrst = 0
    for i in range(len(multi_mcrst)):
        mcrst += multi_mcrst[len(multi_mcrst) - i - 1] * product_prev_sizes(i, intervals)
    return mcrst


def product_prev_sizes(i, intervals):
    prod = 1
    for d in range(i):
        prod *= len(intervals[len(intervals) - 1 - d])
    return prod


# given an index return the related macrostate expressed with a value for each dimension.
def get_mcrst_from_index(index, intervals):
    mcrst = []
    for i in range(len(intervals)):
        prod = product_prev_sizes(len(intervals) - 1 - i, intervals)
        ind = math.floor(index / prod)
        mcrst.append(ind)
        index -= prod*ind
    assert(index == 0)
    return mcrst

# -----------------------------------------------------------------------------------------


def count_actions(container):
    n_actions = 0
    for i in range(0, len(container)):
        n_actions += len(list(container[i].keys()))
    return n_actions


def normalize_array(array):
    den = np.sum(array)
    return array/den


def flat_listoflists(list):
    return [item for sublist in list for item in sublist]


# function used in abstract reward calculation


def calc_abs_reward_lqg(cont, action, Q, R):
    rew = 0
    for act in cont.keys():
        rew += np.dot(cont[act]['state'], np.dot(Q, cont[act]['state'])) + np.dot(action, np.dot(R, action))
    rew = rew.item()
    return - rew / len(cont.items())


# def calc_abs_reward_cartpole(cont, action):
#     return 1 if -0.2093 <= cont[action]['state'] <= 0.2093 else 0


def calc_abs_reward_minigolf(cont, action):
    action = np.clip(action, 0, 5)
    rew = 0
    for act in cont.keys():
        if action < np.sqrt(1.836 * cont[act]['state'][0]):
            rew += -1
        elif action > np.sqrt(7.33 + 1.836 * cont[act]['state'][0]):
            rew += -100
        else:
            rew += 0
    return rew / len(cont.items())


# .................................................


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
    max_mins = mins[0]
    min_maxs = maxs[0]
    for m in mins:
        max_mins = np.maximum(max_mins, m)  # I obtain the max value of mins for each dimension
    for m in maxs:
        min_maxs = np.minimum(min_maxs, m)  # I obtain the min value of maxs for each dimension
    if np.all(np.minimum(max_mins, min_maxs) == max_mins):
        return max_mins, min_maxs
    else:
        return None, None  # void intersection


def build_mcrst_from_samples(samples, n_mcrst, min_val, max_val):
    samples_base = flat_listoflists(samples)
    samples_base = [s[0] for s in samples_base]
    INTERVALS = []

    for i in range(len(n_mcrst)):
        samples = sorted([s[i] for s in samples_base])
        mcrst_size = math.floor(len(samples) / n_mcrst[i])
        dim_int = []

        while len(samples) >= 2 * mcrst_size:
            if len(dim_int) == 0:
                dim_int.append([min_val[i], samples[mcrst_size - 1]])
            else:
                dim_int.append([dim_int[-1][1], samples[mcrst_size - 1]])
            samples = samples[mcrst_size:]
        if len(samples) > 0:
            if len(dim_int) == 0:
                dim_int.append([min_val[i], max_val[i]])
            else:
                dim_int.append([dim_int[-1][1], max_val[i]])
        INTERVALS.append(dim_int)

    return INTERVALS

# INTERVALS = [[-2, -1.95], [-1.95, -1.9], [-1.9, -1.85], [-1.85, -1.8], [-1.8, -1.75], [-1.75, -1.7], [-1.7, -1.65],
#              [-1.65, -1.6], [-1.6, -1.55], [-1.55, -1.5], [-1.5, -1.45], [-1.45, -1.4], [-1.4, -1.35], [-1.35, -1.3],
#              [-1.3, -1.25], [-1.25, -1.2], [-1.2, -1.15], [-1.15, -1.1], [-1.1, -1.05], [-1.05, -1.0], [-1.0, -0.95],
#              [-0.95, -0.9], [-0.9, -0.85], [-0.85, -0.8], [-0.8, -0.75], [-0.75, -0.7], [-0.7, -0.65], [-0.65, -0.6],
#              [-0.6, -0.55], [-0.55, -0.5], [-0.5, -0.45], [-0.45, -0.4], [-0.4, -0.35], [-0.35, -0.3], [-0.3, -0.25],
#              [-0.25, -0.2], [-0.2, -0.15], [-0.15, -0.1], [-0.1, -0.05], [-0.05, -0.015], [-0.015, -0.005],
#               [-5e-3, -5e-6], [-5e-6, -5e-8], [-5e-8, -5e-10],
#               [-5e-10, -5e-12], [-5e-12, -5e-14], [-5e-14, 5e-14], [5e-14, 5e-12], [5e-12, 5e-10],
#               [5e-10, 5e-8], [5e-8, 5e-6], [5e-6, 5e-3],
#               [0.005, 0.015], [0.015, 0.05], [0.05, 0.1],
#              [0.1, 0.15], [0.15, 0.2], [0.2, 0.25], [0.25, 0.3], [0.3, 0.35], [0.35, 0.4], [0.4, 0.45], [0.45, 0.5],
#              [0.5, 0.55], [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 0.75], [0.75, 0.8], [0.8, 0.85], [0.85, 0.9],
#              [0.9, 0.95], [0.95, 1.0], [1.0, 1.05], [1.05, 1.1], [1.1, 1.15], [1.15, 1.2], [1.2, 1.25], [1.25, 1.3],
#              [1.3, 1.35], [1.35, 1.4], [1.4, 1.45], [1.45, 1.5], [1.5, 1.55], [1.55, 1.6], [1.6, 1.65], [1.65, 1.7],
#              [1.7, 1.75], [1.75, 1.8], [1.8, 1.85], [1.85, 1.9], [1.9, 1.95], [1.95, 2.0]]

# INTERVALS = [[-2, -1.6], [-1.6, -1.2], [-1.2, -0.8], [-0.8, -0.5], [-0.5, -0.3],
#              [-0.3, -0.1], [-0.1, -0.05], [-0.05, -0.015], [-0.015, -0.005],
#              [-5e-3, -5e-6], [-5e-6, -5e-8], [-5e-8, -5e-10],
#              [-5e-10, -5e-12], [-5e-12, -5e-14], [-5e-14, 5e-14], [5e-14, 5e-12], [5e-12, 5e-10],
#              [5e-10, 5e-8], [5e-8, 5e-6], [5e-6, 5e-3],
#              [0.005, 0.015], [0.015, 0.05], [0.05, 0.1], [0.1, 0.3], [0.3, 0.5],
#              [0.5, 0.8], [0.8, 1.2], [1.2, 1.6], [1.6, 2]]

# INTERVALS = [[-2, -1], [-1, -0.4], [-0.4, -0.1], [-0.1, 0.1], [0.1, 0.4], [0.4, 1], [1, 2]]


# test
# interv = [[[], []], [[], [], []], [[], [], [], []]]
# mul_mcrst = [1, 1, 3]
# print(get_multidim_mcrst(mul_mcrst, interv))