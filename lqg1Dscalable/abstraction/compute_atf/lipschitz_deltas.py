from lqg1Dscalable.abstraction.lipschitz_abstraction import LipschitzAbstraction
import lqg1Dscalable.abstraction.compute_atf.abstract_tf.sample_distribution as sample_dist
import lqg1Dscalable.helper as helper
import math


class LipschitzDeltaS(LipschitzAbstraction):

    def __init__(self, a, b, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)
        self.LIPSCHITZ_CONST_S = abs(a - 1)
        self.LIPSCHITZ_CONST_A = b

    def calculate_single_atf(self, cont, act, std=0):

        new_state_bounds = []
        delta_s = cont[act]['new_state'] - cont[act]['state']
        # epsilon_bound = 2 * std * math.sqrt(2 / math.pi)

        for action in cont.keys():

            dist_s_shat = abs(cont[act]['state'] - cont[action]['state'])
            # the bound is the difference I can have when I take act in a diff state
            # according to the Lipschitz hypothesis on delta s.
            bound1 = self.LIPSCHITZ_CONST_S * dist_s_shat
            min_val1 = cont[action]['state'] + delta_s - bound1
            max_val1 = cont[action]['state'] + delta_s + bound1

            dist_a_ahat = abs(action - act)
            delta_s2 = cont[action]['new_state'] - cont[action]['state']
            # the bound is the difference I can have when I take act instead of action
            # according to the Lipschitz hypothesis on delta s.
            bound2 = self.LIPSCHITZ_CONST_A * dist_a_ahat
            min_val2 = cont[action]['state'] + delta_s2 - bound2
            max_val2 = cont[action]['state'] + delta_s2 + bound2

            bounds = [[round(min_val1, 3), round(max_val1, 3)], [round(min_val2, 3), round(max_val2, 3)]]

            # for a, sample in cont.items():
            #     delta_s3 = cont[a]['new_state'] - cont[a]['state']
            #     dist_s_shat2 = abs(sample['state'] - cont[action]['state'])
            #     dist_a_ahat2 = abs(a - act)
            #     bound3 = self.LIPSCHITZ_CONST_S * dist_s_shat2 + self.LIPSCHITZ_CONST_A * dist_a_ahat2
            #     min_val3 = cont[action]['state'] + delta_s3 - bound3
            #     max_val3 = cont[action]['state'] + delta_s3 + bound3
            #     bounds.append([round(min_val3, 3), round(max_val3, 3)])

            min_val, max_val = helper.interval_intersection(bounds)

            new_state_bounds.append([min_val, max_val])

        return sample_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
