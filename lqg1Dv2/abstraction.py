import random
import numpy as np

SEED = None
random.seed(SEED)

SAMPLES_IN_MCRST = 2000
RDM_SAMPLES = 500

# when False I use Lipschitz hypothesis to calculate the abstract Transition Function
# (assuming uniform state distribution in every macrostate)
TF_KNOWN = False


class Abstraction(object):

    def __init__(self, n_episodes, n_steps, intervals, a, b):
        super().__init__()
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        # intervals is an array of pairs (s_min, s_max) representing all the macrostates
        self.intervals = intervals
        self.container = self.init_container()
        self.a = a
        self.b = b
        self.LIPSCHITZ_CONST_TF = b

    def init_container(self):
        container = []
        for i in range(0, len(self.intervals)):
            container.append({})
        return container

    def divide_samples(self, samples):
        self.container = self.init_container()
        # container is an array of dictionaries. Every dict follows this configuration:
        # ---------------------------------------------------------------
        # action: abstract_reward, abstract_tf (later), state, new_state
        # ---------------------------------------------------------------
        for s in samples:
            mcrst = get_mcrst(s[0], self.intervals)
            self.container[mcrst][s[1]] = [self.calc_abs_reward(mcrst, s[1]), None, s[0], s[3]]
        # to avoid a slow computation (quadratic on the # of action sampled in each macrostate)
        self.container = [huge_mcrst_correction(cont) if len(cont.keys()) > SAMPLES_IN_MCRST else cont
                          for cont in self.container]
        # at this point I know all the states sampled for every mcrst -> I can calculate the abstract TFs.
        self.calc_abs_tf()

    def get_container(self):
        return self.container

    def calc_abs_reward(self, st, a):
        s_int = self.intervals[st]
        s_mean = (s_int[0] + s_int[1])/2
        return -0.5 * (s_mean * s_mean + a * a)

    # for each action sampled it calculates the abstract TF as a vector of probabilities to end in each mcrst
    def calc_abs_tf(self):
        if TF_KNOWN:
            for cont in self.container:
                for act in cont.keys():
                    abs_tf = self.calc_single_atf(cont, act)
                    # the probability array (abs_tf) is put in the container, in the position [1] of new_state.
                    cont[act][1] = abs_tf
        else:
            for cont in self.container:
                for act in cont.keys():
                    cont[act][1] = self.calc_single_atf_lipschitz(cont, act)

    def calc_single_atf(self, cont, act):
        # every action needs an array (with length = #mcrst) to represent the abstract transition function
        abs_tf = np.zeros(len(self.intervals))
        # cont[a][2] is one of the sampled states.
        # I consider the effect of taking a certain action in every sampled state belonging to the mcrst.
        n_st_effect = [self.a * cont[a][2] + self.b * act for a in cont.keys()]
        for ns in n_st_effect:
            abs_tf[get_mcrst(ns, self.intervals)] += 1
        abs_tf = [p / len(cont.keys()) for p in abs_tf]
        return abs_tf

    def calc_single_atf_lipschitz(self, cont, act):
        abs_tf = np.zeros(len(self.intervals))
        new_states = sorted([[v[3], k] for k, v in cont.items()])
        new_st_min = new_states[0][0]
        new_st_max = new_states[-1][0]
        # I obtain the min and max new state I would get performing action act in the mcrst, according to the samples.
        min_val = new_st_min - self.LIPSCHITZ_CONST_TF * abs(new_states[0][1] - act)
        max_val = new_st_max + self.LIPSCHITZ_CONST_TF * abs(new_states[-1][1] - act)
        min_val_mcrst = get_mcrst(min_val, self.intervals)
        max_val_mcrst = get_mcrst(max_val, self.intervals)
        if min_val_mcrst == max_val_mcrst:
            abs_tf[min_val_mcrst] += 1
        else:
            norm = max_val - min_val
            abs_tf[min_val_mcrst] += (self.intervals[min_val_mcrst][1] - min_val) / norm
            abs_tf[max_val_mcrst] += (max_val - self.intervals[max_val_mcrst][0]) / norm
            for i in range(min_val_mcrst + 1, max_val_mcrst):
                abs_tf[i] += (self.intervals[i][1] - self.intervals[i][0]) / norm
        return abs_tf


def get_tf_known():
    return TF_KNOWN


def huge_mcrst_correction(cont):
    new_cont = {}
    for i in range(0, RDM_SAMPLES):
        rdm = random.randint(0, len(cont.keys()) - 1)
        index = list(cont.keys())[rdm]
        new_cont[index] = cont[index]
    return new_cont


def get_mcrst(state, intervals):
    # in the case of the highest possible state
    if state >= intervals[-1][1]:
        return len(intervals) - 1
    if state <= intervals[0][0]:
        return 0
    index = 0
    for inter in intervals:
        if inter[0] <= state < inter[1]:
            return index
        else:
            index = index + 1

# for a in cont.keys():
#     # dist = distance between the sample related to the specific action act and a sample in the same mcrst.
#     dist = abs(cont[act][2] - cont[a][2])
#     min_val = new_state - LIPSCHITZ_CONST_TF * dist
#     max_val = new_state + LIPSCHITZ_CONST_TF * dist
#     # knowing min & max val I calculate the transition probs (a uniform state dist in the mcrst is supposed).
#     min_val_mcrst = get_mcrst(min_val, self.intervals)
#     max_val_mcrst = get_mcrst(max_val, self.intervals)
#     abs_tf[min_val_mcrst] += self.intervals[min_val_mcrst][1] - min_val
#     abs_tf[max_val_mcrst] += max_val - self.intervals[max_val_mcrst][0]
#     for i in range(min_val_mcrst + 1, max_val_mcrst):
#         abs_tf[i] += self.intervals[i][1] - self.intervals[i][0]

# dist = distance between the sample related to the specific action act and a sample in the same mcrst.
# mcrst = get_mcrst(cont[act][2], self.intervals)
# dist = max(self.intervals[mcrst][1] - cont[act][2], cont[act][2] - self.intervals[mcrst][0])
# min_val = new_state - LIPSCHITZ_CONST_TF * dist
# max_val = new_state + LIPSCHITZ_CONST_TF * dist
# # knowing min & max val I calculate the transition probs (a uniform state dist in the mcrst is supposed).
# min_val_mcrst = get_mcrst(min_val, self.intervals)
# max_val_mcrst = get_mcrst(max_val, self.intervals)
# abs_tf[min_val_mcrst] = self.intervals[min_val_mcrst][1] - min_val
# abs_tf[max_val_mcrst] = max_val - self.intervals[max_val_mcrst][0]
# for i in range(min_val_mcrst + 1, max_val_mcrst):
#     abs_tf[i] = self.intervals[i][1] - self.intervals[i][0]
