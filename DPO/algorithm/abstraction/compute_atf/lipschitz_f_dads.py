from DPO.algorithm.abstraction.lipschitz_abstraction import LipschitzAbstraction
import DPO.algorithm.abstraction.compute_atf.abstract_tf.sample_distribution as sample_dist
# import DPO.visualizer.bounds_visualizer as bvis
import DPO.helper as helper
import logging


class LipschitzFdads(LipschitzAbstraction):

    def __init__(self, lip_state, lip_action, gamma, sink, a, b, intervals=None):
        super().__init__(gamma, sink, intervals)
        self.LIPSCHITZ_CONST_STATE = lip_state
        self.LIPSCHITZ_CONST_ACTION = lip_action
        self.a = a
        self.b = b

    def calculate_single_atf(self, mcrst, act, std=0):

        new_state_bounds = []
        cont = self.container[mcrst]

        # I obtain the min & max new state I would get by performing action act in every state sampled.
        for action in cont.keys():
            bounds = []

            if std == 0:
                min_val1 = cont[action]['new_state'] - self.LIPSCHITZ_CONST_ACTION * abs(action - act)
                max_val1 = cont[action]['new_state'] + self.LIPSCHITZ_CONST_ACTION * abs(action - act)
                bounds.append([round(min_val1, 3), round(max_val1, 3)])

                state_distance = abs(cont[act]['state'] - cont[action]['state'])
                min_val2 = cont[act]['new_state'] - self.LIPSCHITZ_CONST_STATE * state_distance
                max_val2 = cont[act]['new_state'] + self.LIPSCHITZ_CONST_STATE * state_distance
                bounds.append([round(min_val2, 3), round(max_val2, 3)])

            # if the env is stochastic we need to consider the bound generated from every sample.
            # in this way the bias due to the noise is weighted among all the samples.
            # else:
            #     for a in cont.keys():
            #         dist_s_shat = abs(cont[a]['state'] - cont[action]['state'])
            #         dist_a_ahat = abs(a - act)
            #         bound3 = abs(self.LIPSCHITZ_CONST_STATE * dist_s_shat + self.LIPSCHITZ_CONST_ACTION * dist_a_ahat)
            #         min_val3 = cont[a]['new_state'] - bound3
            #         max_val3 = cont[a]['new_state'] + bound3
            #         bounds.append([round(min_val3, 3), round(max_val3, 3)])

            min_val, max_val = helper.interval_intersection(bounds)
            # if abs(min_val1 - min_val) > 0.001 or abs(max_val1 - max_val) > 0.001:
            #     print("here")

            new_state_bounds.append([min_val, max_val])

        # --- LOG ---
        if mcrst == 0 and act == min(list(self.container[0].keys())):
            logging.debug("Bounds related to min action in mcrst 0: ")
            logging.debug(new_state_bounds)
        if mcrst == 0 and act == max(list(self.container[0].keys())):
            logging.debug("Bounds related to max action in mcrst 0: ")
            logging.debug(new_state_bounds)
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

        # it uses the computed bounds to build a bounded abstract tf.
        # return bounded_atf.abstract_tf(self.intervals, new_state_bounds, self.sink)
        # return uni_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
        return sample_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
