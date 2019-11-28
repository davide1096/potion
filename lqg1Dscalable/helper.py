import random
import numpy as np

SEED = 42
random.seed(SEED)

MAX_SAMPLES_IN_MCRST = 60


def big_mcrst_correction(cont):
    new_cont = {}
    for i in range(0, MAX_SAMPLES_IN_MCRST):
        rdm = random.randint(0, len(cont.keys()) - 1)
        index = list(cont.keys())[rdm]
        new_cont[index] = cont[index]
    return new_cont


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
