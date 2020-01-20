import DPO.helper as helper
import numpy as np
from DPO.helper import Helper


class Abstraction(object):

    def __init__(self, gamma, sink, intervals=None, Q=None, R=None, maxa_env=1):
        super().__init__()
        self.intervals = intervals
        self.container = []
        self.gamma = gamma
        self.sink = sink
        self.Q = Q
        self.R = R
        self.maxa_env = maxa_env

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

        for s in samples:
            # every s is an array with this shape: ['state', 'action', 'reward', 'new_state']
            mcrst = helper.get_mcrst(s[0], self.intervals, self.sink)
            mcrst_index = helper.get_multidim_mcrst(mcrst, self.intervals)
            key = len(self.container[mcrst_index].items())
            self.container[mcrst_index][key] = {'state': s[0], 'action': s[1], 'new_state': s[3]}

        # to avoid a slow computation.
        help = Helper(seed)
        self.container = [help.big_mcrst_correction(cont) if len(cont.items()) > helper.MAX_SAMPLES_IN_MCRST else cont
                          for cont in self.container]

        # calculate the abstract reward for every sample.
        if problem == 'lqg1d' or problem == 'mass':
            reward_func = helper.calc_abs_reward_lqg
        # elif problem == 'cartpole1d':
        #     reward_func = helper.calc_abs_reward_cartpole
        elif problem == 'minigolf':
            reward_func = helper.calc_abs_reward_minigolf
        elif problem == 'safety':
            pass  # TODO abstract reward function (also change the line below)
        for cont in self.container:
            for _, v in cont.items():
                # v['abs_reward'] = reward_func(cont, v['action'], self.Q, self.R, self.maxa_env)
                v['abs_reward'] = 0

    def compute_abstract_tf(self):
        pass

