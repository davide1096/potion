from DPO.algorithm.abstraction.lipschitz_abstraction import LipschitzAbstraction
import DPO.algorithm.abstraction.compute_atf.abstract_tf.sample_distribution as sample_dist
import DPO.algorithm.abstraction.compute_atf.abstract_tf.bounded_atf as bounded_atf
import DPO.helper as helper
import numpy as np
import logging
# import DPO.visualizer.bounds_visualizer as bvis
import gym
import potion.envs


class LipschitzDeltaS(LipschitzAbstraction):

    def __init__(self, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)

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
