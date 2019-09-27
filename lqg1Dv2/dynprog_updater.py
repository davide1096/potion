import numpy as np

EPSILON = 0.001


class Updater(object):

    def __init__(self, n_mcrst, gamma):
        super().__init__()
        self.v_function = np.zeros(n_mcrst)
        self.best_policy = [[] for i in range(0, n_mcrst)]
        self.gamma = gamma

    def solve_mdp(self, container):
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
                new_state = container[i][a][1]
                possible_actions[a] = reward + self.gamma * self.v_function[new_state]
            self.best_policy[i], new_v_function[i] = best_actions(possible_actions)
        return new_v_function


def best_actions(possibilities):
    target = max(possibilities.values())
    best_acts = [k for k in possibilities.keys() if possibilities[k] == target]
    return best_acts, target

