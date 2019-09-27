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

    def init_container(self):
        container = []
        for i in range(0, len(self.intervals)):
            container.append({})
        return container

    def divide_samples(self, samples):
        self.container = self.init_container()
        for s in samples:
            mcrst = get_mcrst(s[0], self.intervals)
            self.container[mcrst][s[1]] = [self.calc_abs_reward(mcrst), get_mcrst(s[3], self.intervals)]

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
        rdm_number = random.randint(0, len(self.container[state])-1)
        actions = self.container[state].keys()
        return actions[rdm_number]

    def get_container(self):
        return self.container

    def calc_abs_reward(self, st):
        ns = self.intervals[st]
        return -(abs(ns[0] + ns[1]))/2


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


