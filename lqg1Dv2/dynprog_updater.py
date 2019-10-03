import numpy as np

EPSILON = 0.1


class Updater(object):

    def __init__(self, n_mcrst, gamma):
        super().__init__()
        self.v_function = np.zeros(n_mcrst)
        self.best_policy = [[] for i in range(0, n_mcrst)]
        self.gamma = gamma

    def solve_mdp(self, container):
        # self.v_function = np.zeros(len(self.v_function))
        new_v = self.single_step_update(container)
        while not self.solved(new_v):
            self.v_function = new_v
            new_v = self.single_step_update(container)
        self.v_function = new_v
        return self.best_policy

    def solved(self, new):
        for n, o in zip(new, self.v_function):
            if abs(o - n) > EPSILON:
                return False
        return True

    def single_step_update(self, container):
        new_v_function = np.empty(len(self.v_function))
        for i in range(0, len(self.v_function)):
            possible_actions = {}
            for a in container[i].keys():
                reward = container[i][a][0]
                abs_tf = container[i][a][1]
                # x is the weighted sum of v_functions related to new_mcrst (weighted according to the abs_tf)
                x = sum([new_mcrst_prob * v_fun for new_mcrst_prob, v_fun in zip(abs_tf, self.v_function)])
                possible_actions[a] = reward + x
            self.best_policy[i], new_v_function[i] = best_actions(possible_actions)
        return new_v_function


def best_actions(possibilities):
    target = max(possibilities.values())
    best_acts = [k for k in possibilities.keys() if possibilities[k] == target]
    return best_acts, target

