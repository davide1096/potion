import DPO.algorithm.updater_abstract.bounded_mdp.VI as VI
import numpy as np
import DPO.helper as helper

# to avoid a slow computation.
MAX_ITERATIONS = 500
EPSILON = 0.0001


class IVI(object):

    def __init__(self, gamma, sink, pes, intervals=None, sink_val=None):
        super().__init__()
        self.intervals = intervals
        self.gamma = gamma
        self.sink = sink
        self.pes = pes
        self.v_function_lb = []
        self.v_function_ub = []
        self.best_policy = []
        self.sink_val = sink_val

        if intervals is not None:
            adder = 1 if sink else 0
            shape = [len(i) + adder for i in self.intervals]

            self.v_function_lb = np.zeros(tuple(shape))
            self.v_function_ub = np.zeros(tuple(shape))
            self.best_policy = []
            num = 1
            for s in shape:
                num *= s - adder
            for i in range(num):
                self.best_policy.append([])

            self.v_function_lb[-1] = self.sink_val
            self.v_function_ub[-1] = self.sink_val

    def solve_mdp(self, container, intervals=None):

        if intervals is not None:
            self.intervals = intervals
            adder = 1 if self.sink else 0
            shape = [len(i) + adder for i in intervals]

            self.v_function_lb = np.zeros(tuple(shape))
            self.v_function_ub = np.zeros(tuple(shape))
            self.best_policy = []
            num = 1
            for s in shape:
                num *= s - adder
            for i in range(num):
                self.best_policy.append([])

        n_iterations = 0

        while True:

            n_iterations += 1
            prev_vf_lb = self.v_function_lb.copy()
            prev_vf_ub = self.v_function_ub.copy()

            self.single_step_update(container)
            self.v_function_correction(container)

            if n_iterations >= MAX_ITERATIONS:
                break

            if self.solved(prev_vf_lb, prev_vf_ub):
                break

        return self.best_policy

    # in empty macrostate it prevents no update on value function.
    # The update is pessimistic: the lowest vf seen.
    def v_function_correction(self, container):
        min_vf_lb = np.min(self.v_function_lb[:-1])  # not consider the sink value
        min_vf_ub = np.min(self.v_function_ub[:-1])
        for i, cont in enumerate(container):
            if len(cont.keys()) == 0:  # empty macrostate
                mcrst = helper.get_mcrst_from_index(i, self.intervals)
                self.v_function_lb[tuple(mcrst)] = min_vf_lb
                self.v_function_ub[tuple(mcrst)] = min_vf_ub

    def solved(self, prev_vf_lb, prev_vf_ub):
        diff_lb = abs(self.v_function_lb - prev_vf_lb)
        diff_ub = abs(self.v_function_ub - prev_vf_ub)
        # True if there is no diff between prev and new value greater than EPSILON.
        return not (np.any(diff_lb > EPSILON) or np.any(diff_ub > EPSILON))

    def single_step_update(self, container):

        lb = []
        ub = []

        # for every bound we keep the state index related to it.
        num = 1
        for inter in self.intervals:
            num *= len(inter)

        for i in range(num):
            mcrst = helper.get_mcrst_from_index(i, self.intervals)
            lb.append([self.v_function_lb[tuple(mcrst)], mcrst])
            ub.append([self.v_function_ub[tuple(mcrst)], mcrst])
        if self.sink:
            lb.append([self.sink_val, 'sink'])
            ub.append([self.sink_val, 'sink'])

        # lb is sorted in increasing order, ub is sorted in decreasing order.
        lb = sorted(lb)
        ub = sorted(ub, reverse=True)

        bp = self.best_policy.copy()
        vf_lb = self.v_function_lb.copy()
        vf_ub = self.v_function_ub.copy()

        for i in range(num):
            poss_actions = {}

            for a in container[i].keys():
                min_v = VI.update(self.v_function_lb, [l[1] for l in lb], a, container[i], self.gamma)
                max_v = VI.update(self.v_function_ub, [u[1] for u in ub], a, container[i], self.gamma)
                poss_actions[a] = [min_v, max_v]

            mcrst = helper.get_mcrst_from_index(i, self.intervals)
            bp[i], vf = self.best_actions_pes(poss_actions, mcrst) if self.pes \
                else self.best_actions_opt(poss_actions, mcrst)
            vf_lb[tuple(mcrst)] = vf[0]
            vf_ub[tuple(mcrst)] = vf[1]

        self.best_policy = bp
        self.v_function_lb = vf_lb
        self.v_function_ub = vf_ub

    def best_actions_pes(self, possibilities, mcrst):

        # find the best interval according to the pessimistic definition of MAX.
        if len(possibilities.items()) > 0:
            # targetlb is the max value of the lower bounds in possibilities.
            targetlb = max([v[0] for i, v in possibilities.items()])
            best_acts = [[i, v] for i, v in possibilities.items() if possibilities[i][0] == targetlb]

            # it returns the single action found with its value function interval.
            if len(best_acts) == 1:
                return [best_acts[0][0]], best_acts[0][1]

            # in the case of multiple action it searches the ones that provide the highest upper bound on vf interval.
            targetub = max([v[1][1] for v in best_acts])
            best_acts = [[i, v] for i, v in possibilities.items() if possibilities[i][0] == targetlb and
                         possibilities[i][1] == targetub]
            return [b[0] for b in best_acts], best_acts[0][1]

        # sink state
        else:
            return None, [self.v_function_lb[tuple(mcrst)], self.v_function_ub[tuple(mcrst)]]

    def best_actions_opt(self, possibilities, mcrst):
        # find the best interval according to the pessimistic ">" definition.
        if len(possibilities.items()) > 0:
            # target is the value of the v_function of the macrostate at this iteration.
            targetub = max([v[1] for i, v in possibilities.items()])
            best_acts = [[i, v] for i, v in possibilities.items() if possibilities[i][1] == targetub]
            if len(best_acts) == 1:
                return [best_acts[0][0]], best_acts[0][1]
            targetlb = max([v[1][0] for v in best_acts])
            best_acts = [[i, v] for i, v in possibilities.items() if possibilities[i][0] == targetlb and
                         possibilities[i][1] == targetub]
            return [b[0] for b in best_acts], best_acts[0][1]

        else:
            return None, [self.v_function_lb[tuple(mcrst)], self.v_function_ub[tuple(mcrst)]]





