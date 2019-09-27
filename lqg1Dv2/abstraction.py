import numpy as np
import random

SEED = None
random.seed(SEED)


class Abstraction(object):

    def __init__(self, n_episodes, n_steps, intervals):
        super().__init__()
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        # intervals is an array of pairs (s_min, s_max) representing all the macrostates
        self.intervals = intervals
        self.container = self.init_container()
        # self.abstract_policy = None
        # self.init_policy()

    def init_container(self):
        container = []
        for i in range(0, len(self.intervals)):
            container.append({})
        return container

    # def init_policy(self):
    #     policy = []
    #     for i in range(0, len(self.intervals)):
    #         policy.append([])
    #     self.abstract_policy = policy

    def divide_samples(self, samples):
        self.container = self.init_container()
        for s in samples:
            next_mcrst = get_mcrst(s[3], self.intervals)
            self.container[get_mcrst(s[0], self.intervals)][s[1]] = [self.calc_abs_reward(next_mcrst), next_mcrst]

    # def compute_abstract_policy(self):
    #     for i in range(0, len(self.intervals)):
    #         self.abstract_policy[i] = [[k, 1 / len(self.container[i])] for k in self.container[i].keys()]
        # for i in range(0, len(self.intervals)):
        #     if len(self.abstract_policy[i]) == 0:
        #         self.abstract_policy[i] = [[k, 1/len(self.container[i])] for k in self.container[i].keys()]
        #     else:
        #         prev_len_policy = len(self.abstract_policy[i])
        #         # add with a mean value the new actions
        #         for k in self.container[i].keys():
        #             self.abstract_policy[i].append([k, 1 / prev_len_policy])
        #         # normalize abs_pol
        #         for prob in self.abstract_policy[i]:
        #             prob[1] = prob[1] / (1 + len(self.container[i]) / prev_len_policy)

    def abstract_sampling(self):
        samples_list = []
        for i in range(0, self.n_episodes):
            state = random.randint(0, len(self.intervals) - 1)
            for j in range(0, self.n_steps):
                single_sample, state = self.abstract_step(state)
                samples_list.append(single_sample)
        # random.shuffle(samples_list)
        return samples_list

    def abstract_step(self, state):
        action = self.draw_action_weighted_policy(state)
        # info contains: [reward, new_state]
        info = self.container[state][action]
        return [state, action, info[0], info[1]], info[1]

    def draw_action_weighted_policy(self, state):
        # policy = self.abstract_policy[state]
        rdm_number = random.randint(0, len(self.container[state])-1)
        actions = self.container[state].keys()
        return actions[rdm_number]

    # def get_abstract_policy(self):
    #     return self.abstract_policy
    #
    # def set_abstract_policy(self, policy):
    #     self.abstract_policy = policy

    def get_container(self):
        return self.container

    def calc_abs_reward(self, new_s):
        ns = self.intervals[new_s]
        return (ns[0] + ns[1])/2


def get_mcrst(state, intervals):
    # in the case of the highest possible state
    if state == intervals[-1][1]:
        return len(intervals) - 1
    index = 0
    for inter in intervals:
        if inter[0] <= state < inter[1]:
            return index
        else:
            index = index + 1


