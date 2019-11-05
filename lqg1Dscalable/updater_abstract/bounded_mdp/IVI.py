import numpy as np
import lqg1Dscalable.updater_abstract.bounded_mdp.VI as VI

# to avoid a slow computation.
MAX_ITERATIONS = 50
EPSILON = 0.01


class IVI(object):

    def __init__(self, gamma, sink, pes, intervals=None):
        super().__init__()
        self.intervals = intervals
        self.gamma = gamma
        self.sink = sink
        self.pes = pes
        self.v_function = []
        self.best_policy = []

        if intervals is not None:
            adder = 1 if self.sink else 0
            self.v_function = []
            for i in range(0, len(intervals) + adder):
                self.v_function.append([0, 0])
            self.best_policy = [[] for i in range(0, len(intervals) + adder)]

    def solve_mdp(self, container, intervals=None):

        if intervals is not None:
            adder = 1 if self.sink else 0
            self.v_function = []
            for i in range(0, len(intervals) + adder):
                self.v_function.append([0, 0])
            self.best_policy = [[] for i in range(0, len(intervals) + adder)]
        n_iterations = 0

        while True:

            n_iterations += 1
            prev_vf = self.v_function.copy()

            self.single_step_update(container)

            if n_iterations >= MAX_ITERATIONS:
                break

            if self.solved(prev_vf):
                break

        return self.best_policy

    def solved(self, prev_vf):
        for prev, new in zip(prev_vf, self.v_function):
            if abs(prev[0] - new[0]) > EPSILON or abs(prev[1] - new[1]) > EPSILON:
                return False
        return True

    def single_step_update(self, container):

        lb = []
        ub = []

        # for every bound we keep the state index related to it.
        for i, v in enumerate(self.v_function):
            lb.append([v[0], i])
            ub.append([v[1], i])

        # lb is sorted in increasing order, ub is sorted in decreasing order.
        lb = sorted(lb)
        ub = sorted(ub, reverse=True)

        bp = self.best_policy.copy()
        vf = self.v_function.copy()

        for i in range(0, len(self.v_function)):
            poss_actions = {}

            for a in container[i].keys():
                min_v = VI.update([v[0] for v in self.v_function], [l[1] for l in lb], a, container[i], self.gamma)
                max_v = VI.update([v[1] for v in self.v_function], [u[1] for u in ub], a, container[i], self.gamma)
                poss_actions[a] = [min_v, max_v]

            bp[i], vf[i] = self.best_actions_pes(poss_actions) if self.pes else self.best_actions_opt(poss_actions)

        self.best_policy = bp
        self.v_function = vf

    def best_actions_pes(self, possibilities):

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
            return None, 0

    def best_actions_opt(self, possibilities):
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
            return None, 0






