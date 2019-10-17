import numpy as np
from lqg1Dscalable.abstraction.abstraction import Abstraction
import lqg1Dscalable.helper as helper


class FKnown(Abstraction):

    def __init__(self, a, b, intervals=None):
        super().__init__(intervals)
        self.a = a
        self.b = b

    def calculate_single_atf(self, cont, act):
        abs_tf = np.zeros(len(self.intervals))
        # I consider the effect of taking a certain action in every sampled state belonging to the mcrst.
        n_st_effect = [self.a * cont[action]['state'] + self.b * act for action in cont.keys()]
        for ns in n_st_effect:
            abs_tf[helper.get_mcrst(ns, self.intervals)] += 1
        return helper.normalize_array(abs_tf)
