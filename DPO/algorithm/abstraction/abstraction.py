import DPO.helper as helper
import numpy as np
from DPO.helper import Helper


class Abstraction(object):

    def __init__(self, gamma, sink, intervals=None):
        super().__init__()
        self.intervals = intervals
        self.container = {}
        self.gamma = gamma
        self.sink = sink

    def get_container(self):
        return self.container

    def divide_samples(self, samples, problem, seed, intervals=None):

        if intervals is not None:
            self.intervals = intervals

        # container is an array of dictionaries.
        # Every dict has the actions as key and another dict as value.
        # The second dict has 'state', 'new_state', 'abs_reward', 'abs_tf' as keys.
        self.container = {}
        reward_helper = {}

        for s in samples:
            # every s is an array with this shape: ['state', 'action', 'reward', 'new_state']
            mcrst = helper.get_mcrst(s[0], self.intervals, self.sink)
            mcrst_index = helper.get_index_from_mcrst(mcrst, self.intervals)
            if mcrst_index not in self.container.keys():
                self.container[mcrst_index] = {}
                reward_helper[mcrst_index] = {'val': 0, 'den': 0}
            key = len(self.container[mcrst_index].items())
            self.container[mcrst_index][key] = {'state': s[0], 'action': s[1], 'new_state': s[3]}
            reward_helper[mcrst_index]['val'] += s[2]
            reward_helper[mcrst_index]['den'] += 1

        # to avoid a slow computation.
        help = Helper(seed)
        for i in self.container.keys():
            if len(self.container[i].items()) > helper.MAX_SAMPLES_IN_MCRST:
                self.container[i] = help.big_mcrst_correction(self.container[i])

        if problem == "safety":
            for i in self.container.keys():
                abs_rew = reward_helper[i]['val'] / reward_helper[i]['den']
                for _, v in self.container[i].items():
                    v['abs_reward'] = abs_rew
        elif problem == "minigolf":
            for i in self.container.keys()[1:]:
                for _, v in self.container[i].items():
                    v['abs_reward'] = 0 if i==0 else -1

    def compute_abstract_tf(self):
        pass


