import DPO.helper as helper
import numpy as np
from DPO.helper import Helper


class Abstraction(object):

    def __init__(self, gamma, sink, intervals=None, Q=None, R=None):
        super().__init__()
        self.intervals = intervals
        self.container = []
        self.gamma = gamma
        self.sink = sink
        self.Q = Q
        self.R = R

    def init_container(self):
        shape = [len(i) for i in self.intervals]

        container = []
        num = 1
        for i in range(len(shape)):
            num *= shape[i]
        for j in range(num):
            container.append({})
        return container

    def get_container(self):
        return self.container

    def divide_samples(self, samples, problem, seed, intervals=None):

        if intervals is not None:
            self.intervals = intervals

        # container is an array of dictionaries.
        # Every dict has the actions as key and another dict as value.
        # The second dict has 'state', 'new_state', 'abs_reward', 'abs_tf' as keys.
        self.container = self.init_container()

        for sam in samples:
            for i, s in enumerate(sam):
                # every s is an array with this shape: ['state', 'action', 'reward', 'new_state']
                mcrst = helper.get_mcrst(s[0], self.intervals, self.sink)
                for m, i in zip(mcrst, self.intervals):
                    assert (0 <= m <= len(i) - 1)
                mcrst_index = helper.get_multidim_mcrst(mcrst, self.intervals)
                self.container[mcrst_index][s[1]] = {'state': s[0], 'new_state': s[3]}

        # to avoid a slow computation.
        help = Helper(seed)
        self.container = [help.big_mcrst_correction(cont) if len(cont.items()) > helper.MAX_SAMPLES_IN_MCRST else cont
                          for cont in self.container]

        # calculate the abstract reward for every sample.
        if problem == 'lqg1d':
            reward_func = helper.calc_abs_reward_lqg
        # elif problem == 'cartpole1d':
        #     reward_func = helper.calc_abs_reward_cartpole
        elif problem == 'minigolf':
            reward_func = helper.calc_abs_reward_minigolf
        for cont in self.container:
            for act in cont.keys():
                cont[act]['abs_reward'] = reward_func(cont, act, self.Q, self.R)

    def compute_abstract_tf(self):
        pass

