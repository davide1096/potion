import numpy as np
import DPO.helper as helper

# to avoid a slow computation.
MAX_ITERATIONS = 500
EPSILON = 0.0001


class AbsUpdater(object):

    def __init__(self, gamma, sink, intervals=None, sink_val=None):
        super().__init__()
        self.intervals = intervals
        self.gamma = gamma
        self.sink = sink
        self.v_function = []
        self.best_policy = []
        self.sink_val = sink_val

        if intervals is not None:
            adder = 1 if sink else 0
            shape = [len(i) + adder for i in self.intervals]

            self.v_function = np.zeros(tuple(shape))
            self.best_policy = []
            num = 1
            for s in shape:
                num *= s - adder
            for i in range(num):
                self.best_policy.append([])

    def solve_mdp(self, container, intervals=None):

        if intervals is not None:
            self.intervals = intervals
            adder = 1 if self.sink else 0
            shape = [len(i) + adder for i in intervals]

            self.v_function = np.zeros(tuple(shape))
            self.best_policy = []
            num = 1
            for s in shape:
                num *= s - adder
            for i in range(num):
                self.best_policy.append([])

        new_vf = self.single_step_update(container)
        n_iterations = 0

        while not self.solved(new_vf) and n_iterations < MAX_ITERATIONS:
            self.v_function = new_vf
            new_vf = self.single_step_update(container)
            n_iterations += 1

        self.v_function = new_vf
        return self.best_policy

    def solved(self, new):
        diff = abs(self.v_function - new)
        # True if there is no diff between prev and new value greater than EPSILON.
        return not np.any(diff > EPSILON)

    def single_step_update(self, container):

        new_v_function = np.empty_like(self.v_function)
        for i in range(len(container)):
            possible_actions = {}

            for a in container[i].keys():
                abs_reward = container[i][a]['abs_reward']

                if 'abs_tf' in container[i][a]:
                    abs_tf = container[i][a]['abs_tf']
                    # x is the sum of the v_functions of new_mcrst, weighted according to the abs_tf.
                    x = np.sum(abs_tf * self.v_function)
                    possible_actions[a] = abs_reward + self.gamma * x

            mcrst = helper.get_mcrst_from_index(i, self.intervals)
            self.best_policy[i], new_v_function[tuple(mcrst)] = self.best_actions(possible_actions, i)

        return new_v_function

    def best_actions(self, possibilities, i):

        if len(possibilities.items()) > 0:
            # target is the value of the v_function of the macrostate at this iteration.
            target = max(possibilities.values())
            best_acts = [k for k in possibilities.keys() if possibilities[k] == target]
            return best_acts, target

        else:
            mcrst = helper.get_mcrst_from_index(i, self.intervals)
            return None, self.v_function[tuple(mcrst)]
