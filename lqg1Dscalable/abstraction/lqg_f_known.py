import numpy as np
from lqg1Dscalable.abstraction.abstraction import Abstraction
import lqg1Dscalable.helper as helper
import lqg1Dscalable.abstraction.abstract_tf.sample_distribution as sample_dist
import lqg1Dscalable.abstraction.abstract_tf.uniform_state_distribution as uni_dist


class LqgFKnown(Abstraction):

    def __init__(self, a, b, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)
        self.a = a
        self.b = b

    def calculate_single_atf(self, cont, act):

        new_state_bounds = []
        # I consider the effect of taking a certain action in every sampled state belonging to the mcrst.
        for action in cont.keys():
            new_state = self.a * cont[action]['state'] + self.b * act
            new_state_bounds.append([new_state, new_state])

        return sample_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
        # return uni_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
