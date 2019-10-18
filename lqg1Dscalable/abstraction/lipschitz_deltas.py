import numpy as np
from lqg1Dscalable.abstraction.abstraction import Abstraction
import lqg1Dscalable.helper as helper
import lqg1Dscalable.abstraction.abstract_tf.uniform_state_distribution as uni_dist
import lqg1Dscalable.abstraction.abstract_tf.sample_distribution as sample_dist


class LipschitzDeltaS(Abstraction):

    def __init__(self, lipschitz_st, lipschitz_act, gamma, intervals=None):
        super().__init__(gamma, intervals)
        self.LIPSCHITZ_CONST_S = lipschitz_st
        self.LIPSCHITZ_CONST_A = lipschitz_act

    def calculate_single_atf(self, cont, act):

        new_state_bounds = []
        delta_s = cont[act]['new_state'] - cont[act]['state']

        for action in cont.keys():
            dist_s_shat = abs(cont[act]['state'] - cont[action]['state'])
            # the bound is the difference I can have when I take act in a diff state
            # according to the Lipschitz hypothesis on delta s.
            bound = self.LIPSCHITZ_CONST_A * abs(act - act) + self.LIPSCHITZ_CONST_S * dist_s_shat

            # It works for both positive and negative delta_s.
            min_val = cont[action]['state'] + delta_s - bound
            max_val = cont[action]['state'] + delta_s + bound
            new_state_bounds.append([min_val, max_val])

        return sample_dist.abstract_tf(self.intervals, new_state_bounds)
