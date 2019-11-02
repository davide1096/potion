import numpy as np

# to avoid a slow computation.
MAX_ITERATIONS = 50
EPSILON = 0.1


class AbsStepUpdater(object):

    def __init__(self, gamma, sink, intervals=None):
        super().__init__()
        self.intervals = intervals
        self.gamma = gamma
        self.sink = sink
        self.v_function = []
        self.best_policy = []

        if intervals is not None:
            adder = 1 if self.sink else 0
            self.v_function = np.zeros(len(intervals) + adder)
            self.best_policy = [[] for i in range(0, len(intervals) + adder)]

    def solve_mdp(self, container, intervals=None):

        if intervals is not None:
            adder = 1 if self.sink else 0
            self.v_function = np.zeros(len(intervals) + adder)
            self.best_policy = [[] for i in range(0, len(intervals) + adder)]

        empty = True
        for bp in self.best_policy:
            if len(bp) != 0:
                empty = False

        if empty:

            new_bp, new_vf = self.single_step_update(container)
            n_iterations = 0

            while not self.solved(new_vf) and n_iterations < MAX_ITERATIONS:
                self.v_function = new_vf
                self.best_policy = new_bp
                new_bp, new_vf = self.single_step_update(container)
                n_iterations += 1

            return self.best_policy, True

        else:

            new_bp, new_vf = self.single_step_update(container)
            upd = False
            if self.worth_update(new_vf):
                self.v_function = new_vf
                self.best_policy = new_bp
                upd = True

            return self.best_policy, upd

    def solved(self, new):
        for n, o in zip(new, self.v_function):
            if abs(o - n) > EPSILON:
                return False
        return True

    def worth_update(self, new):
        for n, o in zip(new, self.v_function):
            if o - n > EPSILON:
                return False
        return True

    def single_step_update(self, container):

        new_v_function = np.empty(len(self.v_function))
        new_best_policy = [[] for i in range(0, len(self.v_function))]
        for i in range(0, len(self.v_function)):
            possible_actions = {}

            for a in container[i].keys():
                abs_reward = container[i][a]['abs_reward']
                abs_tf = container[i][a]['abs_tf']
                # x is the sum of the v_functions of new_mcrst, weighted according to the abs_tf.
                x = sum([new_mcrst_prob * v_fun for new_mcrst_prob, v_fun in zip(abs_tf, self.v_function)])
                possible_actions[a] = abs_reward + self.gamma * x

            new_best_policy[i], new_v_function[i] = self.best_actions(possible_actions)

        return new_best_policy, new_v_function

    def best_actions(self, possibilities):

        if len(possibilities.items()) > 0:
            # target is the value of the v_function of the macrostate at this iteration.
            target = max(possibilities.values())
            best_acts = [k for k in possibilities.keys() if possibilities[k] == target]
            return best_acts, target

        else:
            return None, min(self.v_function)
