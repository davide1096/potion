from lqg1Dscalable.abstraction.lipschitz_abstraction import LipschitzAbstraction
import lqg1Dscalable.abstraction.compute_atf.abstract_tf.sample_distribution as sample_dist


class LqgFKnown(LipschitzAbstraction):

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
