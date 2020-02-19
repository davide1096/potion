from DPO.algorithm.abstraction.abstraction import Abstraction
import DPO.algorithm.abstraction.compute_atf.sample_distribution as sample_dist
import DPO.algorithm.abstraction.compute_atf.bounded_atf as bounded_atf


class LipschitzDeltaS(Abstraction):

    def __init__(self, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)

    def compute_abstract_tf(self, Lds=0):
        for k1, v1 in self.container.items():
            for k2, v2 in v1.items():
                v2['abs_tf'] = self.calculate_single_atf(k1, k2, Lds)

    # ds0 is True when the hypothesis of deltaS = 0 is valid.
    # It means that taking the same action in different states will produce the same delta s (deltas = s' - s).
    def calculate_single_atf(self, k1, k2, Lds=0):

        # k1 index of the mcrst
        # k2 index of the action

        cont = self.container[k1]
        new_states = []
        delta_s = cont[k2]['new_state'] - cont[k2]['state']

        for k, v in cont.items():

            ns = cont[k]['state'] + delta_s
            if Lds == 0:
                new_states.append(ns)
            else:
                dist_s = abs(cont[k]['state'] - cont[k2]['state'])
                new_states.append([ns - Lds * dist_s, ns + Lds * dist_s])

        if Lds == 0:
            return sample_dist.abstract_tf(self.intervals, new_states, self.sink)
        else:
            return bounded_atf.abstract_tf(self.intervals, new_states, self.sink)