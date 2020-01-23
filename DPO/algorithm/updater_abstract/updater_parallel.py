import numpy as np
import DPO.helper as helper
import multiprocessing as mp
import os

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
        self.results = []

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

    def collect_result(self, result):
        self.results.append(result)

    def single_step_update(self, container):

        new_v_function = np.empty_like(self.v_function)
        pool = mp.Pool(len(os.sched_getaffinity(0)))

        self.results = []
        for i, cont in enumerate(container):
            pool.apply_async(self.single_step_update_parallel, args=(i, cont, self.v_function.copy(), self.gamma),
                             callback=self.collect_result)

        pool.close()
        pool.join()

        self.results.sort(key=lambda x: x[0])
        nvf = [r[1] for i, r in self.results]
        self.best_policy = [r[0] for i, r in self.results]

        for i, single_vf in enumerate(nvf):
            mcrst = helper.get_mcrst_from_index(i, self.intervals)
            new_v_function[tuple(mcrst)] = single_vf

        return new_v_function

    def single_step_update_parallel(self, i, cont, vf_self, gamma):

        possible_actions = {}

        for a in cont.keys():
            abs_reward = cont[a]['abs_reward']

            if 'abs_tf' in cont[a]:
                abs_tf = cont[a]['abs_tf']
                # x is the sum of the v_functions of new_mcrst, weighted according to the abs_tf.
                x = np.sum(abs_tf * vf_self)
                possible_actions[a] = abs_reward + gamma * x

        bp, nvf = self.best_actions(possible_actions, i)

        return (i, [bp, nvf])

    def best_actions(self, possibilities, i):

        if len(possibilities.items()) > 0:
            # target is the value of the v_function of the macrostate at this iteration.
            target = max(possibilities.values())
            best_acts = [k for k in possibilities.keys() if possibilities[k] == target]
            return best_acts, target

        else:
            mcrst = helper.get_mcrst_from_index(i, self.intervals)
            return None, self.v_function[tuple(mcrst)]
