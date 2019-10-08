import numpy as np
from lqg1Dv2.abstraction import get_mcrst

EPSILON = 0.01
# alternative way of calculating the V function, it allows to avoid auto-rings in the abstract MDP.
# it is only implemented in the following case:
# 1) deterministic environment
# 2) the transition function in the deterministic environment is known
# 3) no abstract transition function, the effect of an action is the one related to the sample
MULTIACTION = False


class Updater(object):

    def __init__(self, intervals, gamma):
        super().__init__()
        self.intervals = intervals
        self.v_function = np.zeros(len(intervals))
        self.best_policy = [[] for i in range(0, len(intervals))]
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
            if not MULTIACTION:
                for a in container[i].keys():
                    reward = container[i][a][0]
                    abs_tf = container[i][a][1]
                    # x is the weighted sum of v_functions related to new_mcrst (weighted according to the abs_tf)
                    x = sum([new_mcrst_prob * v_fun for new_mcrst_prob, v_fun in zip(abs_tf, self.v_function)])
                    possible_actions[a] = reward + self.gamma * x
            else:
                for a in container[i].keys():
                    possible_actions[a] = self.manage_multiaction(container[i][a], a, mean_reward(container))
            self.best_policy[i], new_v_function[i] = best_actions(possible_actions)
        return new_v_function

    def manage_multiaction(self, action_tuple, action, mean_reward):
        reward = action_tuple[0]
        state = action_tuple[2]
        new_state = action_tuple[3]
        if get_mcrst(state, self.intervals) != get_mcrst(new_state, self.intervals):
            return (reward - mean_reward) + self.gamma * self.v_function[get_mcrst(new_state, self.intervals)]
        else:
            # the new  state is in the same mcrst of the prev state
            acc = reward - mean_reward
            gam = self.gamma
            while get_mcrst(state, self.intervals) == get_mcrst(new_state, self.intervals):
                acc += (reward - mean_reward) * gam
                gam *= self.gamma
                new_state = new_state + action
            return acc + gam * self.v_function[get_mcrst(new_state, self.intervals)]


def best_actions(possibilities):
    target = max(possibilities.values())
    best_acts = [k for k in possibilities.keys() if possibilities[k] == target]
    return best_acts, target


def mean_reward(container):
    acc = 0
    i = 0
    for cont in container:
        for a in cont.keys():
            acc += cont[a][0]
            i += 1
    return acc / i

