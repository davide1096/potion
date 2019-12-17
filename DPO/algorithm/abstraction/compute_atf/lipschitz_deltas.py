from DPO.algorithm.abstraction.lipschitz_abstraction import LipschitzAbstraction
import DPO.algorithm.abstraction.compute_atf.abstract_tf.sample_distribution as sample_dist
import DPO.algorithm.abstraction.compute_atf.abstract_tf.bounded_atf as bounded_atf
import DPO.helper as helper
import logging
# import DPO.visualizer.bounds_visualizer as bvis


class LipschitzDeltaS(LipschitzAbstraction):

    def __init__(self, gamma, sink, intervals=None, a=None, b=None):
        super().__init__(gamma, sink, intervals)
        if a is not None:
            self.LIPSCHITZ_CONST_S = abs(a - 1)
            self.a = a
        if b is not None:
            self.LIPSCHITZ_CONST_A = b
            self.b = b

    def calculate_single_atf(self, mcrst, act, optA, std=0):

        if optA:
            self.LIPSCHITZ_CONST_S = 0

        cont = self.container[mcrst]
        new_state_bounds = []
        delta_s = cont[act]['new_state'] - cont[act]['state']

        for action in cont.keys():

            dist_s_shat = abs(cont[act]['state'] - cont[action]['state'])
            # the bound is the difference I can have when I take act in a diff state
            # according to the Lipschitz hypothesis on delta s.
            bound1 = self.LIPSCHITZ_CONST_S * dist_s_shat
            min_val1 = cont[action]['state'] + delta_s - bound1
            max_val1 = cont[action]['state'] + delta_s + bound1

            bounds = [[round(min_val1, 3), round(max_val1, 3)]]

            if not optA:
                dist_a_ahat = abs(action - act)
                delta_s2 = cont[action]['new_state'] - cont[action]['state']
                # the bound is the difference I can have when I take act instead of action
                # according to the Lipschitz hypothesis on delta s.
                bound2 = self.LIPSCHITZ_CONST_A * dist_a_ahat
                min_val2 = cont[action]['state'] + delta_s2 - bound2
                max_val2 = cont[action]['state'] + delta_s2 + bound2

                bounds.append([round(min_val2, 3), round(max_val2, 3)])

            min_val, max_val = helper.interval_intersection(bounds)
            if min_val is not None and max_val is not None:
                new_state_bounds.append([min_val, max_val])
            # use it only when you need to plot bounds
            # else:
            #     new_state_bounds.append([0, 0])

        # --- LOG ---
        # if mcrst == 0 and act == min(list(self.container[0].keys())):
        #     logging.debug("Bounds related to min action in mcrst 0: ")
        #     logging.debug(new_state_bounds)
        # if mcrst == 0 and act == max(list(self.container[0].keys())):
        #     logging.debug("Bounds related to max action in mcrst 0: ")
        #     logging.debug(new_state_bounds)
        # -----------

        # --- matplot ---
        # if mcrst == 0 and act == min(list(self.container[0].keys())):
        #     true_value = []
        #     for action in cont.keys():
        #         true_value.append(self.a * cont[action]['state'] + self.b * act)
        #     bvis.plot_bounds(new_state_bounds, "min action", true_value)
        # if mcrst == 0 and act == max(list(self.container[0].keys())):
        #     true_value = []
        #     for action in cont.keys():
        #         true_value.append(self.a * cont[action]['state'] + self.b * act)
        #     bvis.plot_bounds(new_state_bounds, "max action", true_value)
        # ---------------

        if optA:
            return sample_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
        else:
            return bounded_atf.abstract_tf(self.intervals, new_state_bounds, self.sink)
