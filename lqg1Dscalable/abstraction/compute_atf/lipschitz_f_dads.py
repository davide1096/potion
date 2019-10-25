from lqg1Dscalable.abstraction.lipschitz_abstraction import LipschitzAbstraction
import lqg1Dscalable.abstraction.compute_atf.abstract_tf.uniform_state_distribution as uni_dist
import lqg1Dscalable.abstraction.compute_atf.abstract_tf.sample_distribution as sample_dist
import lqg1Dscalable.helper as helper


class LipschitzFdads(LipschitzAbstraction):

    def __init__(self, lip_state, lip_action, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)
        self.LIPSCHITZ_CONST_STATE = lip_state
        self.LIPSCHITZ_CONST_ACTION = lip_action

    def calculate_single_atf(self, cont, act, std=0):

        new_state_bounds = []

        # I obtain the min & max new state I would get by performing action act in every state sampled.
        for action in cont.keys():
            min_val1 = cont[action]['new_state'] - self.LIPSCHITZ_CONST_ACTION * abs(action - act)
            max_val1 = cont[action]['new_state'] + self.LIPSCHITZ_CONST_ACTION * abs(action - act)

            state_distance = abs(cont[act]['state'] - cont[action]['state'])
            min_val2 = cont[act]['new_state'] - self.LIPSCHITZ_CONST_STATE * state_distance
            max_val2 = cont[act]['new_state'] + self.LIPSCHITZ_CONST_STATE * state_distance

            bounds = [[round(min_val1, 3), round(max_val1, 3)], [round(min_val2, 3), round(max_val2, 3)]]
            min_val, max_val = helper.interval_intersection(bounds)

            new_state_bounds.append([min_val, max_val])

        return uni_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
        # return sample_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
