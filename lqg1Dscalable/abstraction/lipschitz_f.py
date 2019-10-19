import numpy as np
from lqg1Dscalable.abstraction.abstraction import Abstraction
import lqg1Dscalable.helper as helper
import lqg1Dscalable.abstraction.abstract_tf.uniform_state_distribution as uni_dist
import lqg1Dscalable.abstraction.abstract_tf.sample_distribution as sample_dist


class LipschitzF(Abstraction):

    def __init__(self, lipschitz, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)
        self.LIPSCHITZ_CONST_F = lipschitz

    def calculate_single_atf(self, cont, act):

        new_state_bounds = []
        for action in cont.keys():
            # I obtain the min & max new state I would get by performing action act in every state sampled.
            min_val = cont[action]['new_state'] - self.LIPSCHITZ_CONST_F * abs(action - act)
            max_val = cont[action]['new_state'] + self.LIPSCHITZ_CONST_F * abs(action - act)
            new_state_bounds.append([min_val, max_val])

        return uni_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
        # return sample_dist.abstract_tf(self.intervals, new_state_bounds, self.sink) # NOT WORKING!
