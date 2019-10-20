import random

SEED = 42
random.seed(SEED)

MAX_SAMPLES_IN_MCRST = 30


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


def estimate_J_lqg(samples, gamma):
    acc = 0
    for sam in samples:
        g = 1
        for s in sam:
            # sum of discounted rewards of the initial samples.
            acc += g * s[2]
            g *= gamma
    return acc / len(samples)


def calc_abs_reward_cartpole(cont, action):
    return 1 if -0.2093 <= cont[action]['state'] <= 0.2093 else 0


def estimate_J_cartpole(fict_samples, gamma):
    acc = 0
    for sample in fict_samples:
        g = 1
        for s in sample:
            acc += g
            g *= gamma
    return acc / len(fict_samples)
