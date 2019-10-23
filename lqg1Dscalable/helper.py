import random
import numpy as np

SEED = 42
random.seed(SEED)
MAX_CARTPOLE_THETA = 0.2093
MIN_CARTPOLE_THETA = -0.2093

MAX_SAMPLES_IN_MCRST = 40


def big_mcrst_correction(cont):
    new_cont = {}
    for i in range(0, MAX_SAMPLES_IN_MCRST):
        rdm = random.randint(0, len(cont.keys()) - 1)
        index = list(cont.keys())[rdm]
        new_cont[index] = cont[index]
    return new_cont


def get_mcrst(state, intervals, sink):

    # len(intervals) is the number of dimensions.
    macrostate = np.empty(len(intervals))

    for i, dim_interval in enumerate(intervals):

        # if we have to consider the sink state.
        if sink and state[i] < dim_interval[0][0]:
            macrostate[i] = -1
        elif sink and state[i] > dim_interval[-1][1]:
            macrostate[i] = len(dim_interval)

        # in absence of a sink state.
        elif state[i] >= dim_interval[-1][1]:             # above the upper bound of the state space
            macrostate[i] = len(dim_interval) - 1
        elif state[i] <= dim_interval[0][0]:              # below the lower bound of the state space
            macrostate[i] = 0

        else:                                              # inside the state space
            index = 0
            for inter in dim_interval:
                if inter[0] <= state[i] < inter[1]:
                    macrostate[i] = index
                else:
                    index += 1

    return macrostate


def calc_abs_reward_lqg(cont, action):
    rew = 0
    for act in cont.keys():
        rew += -0.5 * (cont[act]['state'] * cont[act]['state'] + action * action)
    return rew / len(cont.items())


def calc_abs_reward_lqg2d():
    # TODO
    pass


def calc_abs_reward_cartpole(cont, action):
    return 1 if MIN_CARTPOLE_THETA <= cont[action]['state'] <= MAX_CARTPOLE_THETA else 0


def count_actions(container):
    return sum([len(cont.item()) for cont in container])


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


# different from the previous function because I need to pass the fictitious sample for cartpole
# they don't have any action in the sink state (added for a correct value iteration).
# they don't have the reward in the structure of the sample.
def estimate_J_cartpole(fict_samples, gamma):
    acc = 0
    for sample in fict_samples:
        g = 1
        for s in sample:
            acc += g
            g *= gamma
    return acc / len(fict_samples)
