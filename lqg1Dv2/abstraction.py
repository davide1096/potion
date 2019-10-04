import random
import numpy as np
from lqg1Dv2.noisy_env import NoisyEnvironment

SEED = None
random.seed(SEED)

SAMPLES_IN_MCRST = 50
RDM_SAMPLES = 50


class Abstraction(object):

    def __init__(self, n_episodes, n_steps, intervals, env_noise):
        super().__init__()
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        # intervals is an array of pairs (s_min, s_max) representing all the macrostates
        self.intervals = intervals
        self.env_noise = env_noise
        self.noisy_env_helper = NoisyEnvironment(intervals)
        self.container = self.init_container()

    def init_container(self):
        container = []
        for i in range(0, len(self.intervals)):
            container.append({})
        return container

    def divide_samples(self, samples):
        self.container = self.init_container()
        # container is an array of dictionaries. Every dict follows this configuration:
        # ---------------------------------------------------------
        # action: abstract_reward, new_state abstract probabilities (later), state
        # ---------------------------------------------------------
        for s in samples:
            mcrst = get_mcrst(s[0], self.intervals)
            self.container[mcrst][s[1]] = [self.calc_abs_reward(mcrst, s[1]), None, s[0]]
        # to avoid a slow computation (quadratic on the # of action sampled in each macrostate)
        self.container = [huge_mcrst_correction(cont) if len(cont.keys()) > SAMPLES_IN_MCRST else cont
                          for cont in self.container]
        # at this point I have all the states sampled for every mcrst -> I can calculate the abstract TFs.
        self.calc_abs_tf()

    def get_container(self):
        return self.container

    def calc_abs_reward(self, st, a):
        s_int = self.intervals[st]
        s_mean = (s_int[0] + s_int[1])/2
        return -0.5 * (s_mean * s_mean + a * a)

    # for each action sampled it calculates the abstract TF as a vector of probabilities to end in each mcrst
    def calc_abs_tf(self):
        for cont in self.container:
            for act in cont.keys():
                abs_tf = self.calc_single_atf(cont, act)
                # the probability array (abs_tf) is put in the container, in the position [1] of new_state.
                cont[act][1] = abs_tf

    def calc_single_atf(self, cont, act):
        # every action needs an array (with length = #mcrst) to represent the abstract transition function.
        abs_tf = np.zeros(len(self.intervals))
        # calculate the distribution on the arriving mcrsts given act, for every state sampled in a certain mcrst.
        for a in cont.keys():
            # the noise is a std gaussian to be summed to the supposed new state.
            # mu = the supposed new_state, sigma = env_noise const; (cont[a][2] is one of the sampled states).
            new = self.noisy_env_helper.get_mcrst_prob(cont[a][2] + act, self.env_noise)
            abs_tf = [acc + n for acc, n in zip(abs_tf, new)]
        den = sum(abs_tf)
        abs_tf = [p / den for p in abs_tf]
        return abs_tf

    def show_abstract_mdp(self):
        for i in range(0, len(self.intervals)):
            print("\n--------------------------------------")
            print("Macrostate {} whit interval [{}, {}]".format(i, self.intervals[i][0], self.intervals[i][1]))
            print("--------------------------------------")
            [print("{0:.5f}".format(key), " :: ", show_helper(value)) for (key, value)
             in sorted(self.container[i].items())]


def huge_mcrst_correction(cont):
    new_cont = {}
    for i in range(0, RDM_SAMPLES):
        rdm = random.randint(0, len(cont.keys()) - 1)
        index = list(cont.keys())[rdm]
        new_cont[index] = cont[index]
    return new_cont


def get_mcrst(state, intervals):
    # in the case of the highest possible state
    if state == intervals[-1][1]:
        return len(intervals) - 1
    index = 0
    for inter in intervals:
        if inter[0] <= state < inter[1]:
            return index
        else:
            index = index + 1


def show_helper(value):
    rew_msg = "Reward: {0:.5f}\n".format(value[0])
    tf_msg = "TF: "
    for i in range(0, len(value[1])):
        prob = value[1][i]
        if prob >= 0.01:
            tf_msg += "to {0:.0f}: ".format(i) + "{0:.2f}; ".format(prob)
    return rew_msg + tf_msg + "\n"


