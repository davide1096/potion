import numpy as np
from lqg1Dscalable.abstraction.abstraction import Abstraction
import lqg1Dscalable.helper as helper


class LipschitzF(Abstraction):

    def __init__(self, lipschitz, intervals=None):
        super().__init__(intervals)
        self.LIPSCHITZ_CONST_F = lipschitz

    def calculate_single_atf(self, cont, act):
        abs_tf = np.zeros(len(self.intervals))
        new_states = sorted([[v['new_state'], k] for k, v in cont.items()])
        new_st_min = new_states[0][0]
        new_st_max = new_states[-1][0]

        # I obtain the min & max new state I would get by performing action act in the mcrst, according to the samples.
        min_val = new_st_min - self.LIPSCHITZ_CONST_F * abs(new_states[0][1] - act)
        max_val = new_st_max + self.LIPSCHITZ_CONST_F * abs(new_states[-1][1] - act)
        min_val_mcrst = helper.get_mcrst(min_val, self.intervals)
        max_val_mcrst = helper.get_mcrst(max_val, self.intervals)

        if min_val_mcrst == max_val_mcrst:
            abs_tf[min_val_mcrst] += 1

        else:
            abs_tf[min_val_mcrst] += (self.intervals[min_val_mcrst][1] - min_val)
            abs_tf[max_val_mcrst] += (max_val - self.intervals[max_val_mcrst][0])
            for i in range(min_val_mcrst + 1, max_val_mcrst):
                abs_tf[i] += (self.intervals[i][1] - self.intervals[i][0])

        return helper.normalize_array(abs_tf)
