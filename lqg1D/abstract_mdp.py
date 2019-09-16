from gym.utils import seeding
import torch
import random
import lqg1D.lqgspo as lqgspo
import numpy as np
import scipy.stats as stats

SEED = 42


# since we have the parameters related to the abstract transition functions we proceed in this way:
# 1) calculate p(x'|x, a) for all x'
# 2) divide the space [0,1) according to these probabilities
# 3) select the macrostate related to the space that contains the rdm_number
def calculate_new_state(w_x, ac, rdm_number):
    if rdm_number == 0:
        return 0
    den = np.sum(np.exp(w_x * ac))
    prob = [np.exp(w_xi * ac)/den for w_xi in w_x]
    index = 0
    ref = 0
    while rdm_number > ref:
        ref += prob[index]
        index += 1
    return index - 1


def count_states(states):
    unique, counts = np.unique(states, return_counts=True)
    return dict(zip(unique, counts))


class AbstractMdp(object):

    def __init__(self, functions, min_action, max_action):
        super().__init__()
        # here abstract policy, abstract transition and reward functions are contained
        self.functions = functions
        self.np_random, seed = seeding.np_random(SEED)
        self.mcrst_intervals = functions.get_mcrst_intervals()
        # print("The abstract MDP will have the following macrostates: {}".format(self.mcrst_intervals))
        self.state = None
        self.reset()
        self.min_action = min_action
        self.max_action = max_action

    def step(self):
        # draw an action according to the stochastic policy defined for the current macrostate
        par = self.functions.get_policy_parameters(self.state)
        mu = next(par).detach()
        sigma = np.exp(next(par).detach())
        a = stats.truncnorm((self.min_action - mu)/sigma, (self.max_action - mu)/sigma, loc=mu, scale=sigma).rvs()
        # a = self.np_random.normal(loc=next(par).detach(), scale=torch.exp(next(par).detach()))

        # calculate reward and new state according to the sampled action
        r = lqgspo.abstract_reward_function(self.mcrst_intervals[self.state], a)
        new_s = calculate_new_state(self.functions.get_tf_parameters(self.state), a, self.np_random.uniform(0, 1))

        # print("Sample from the abstract MDP: S:{}, A:{}, R:{}, S':{}".format(self.state, a, r, new_s))
        state = self.state
        self.state = new_s
        return [state, a, r, new_s]

    def reset(self):
        self.state = int(self.np_random.uniform(low=0, high=len(self.mcrst_intervals)))

    def sampling(self, n_samples, n_steps):
        samples_list = []
        for i in range(0, n_samples):
            self.reset()

            for j in range(0, n_steps):
                samples_list.append(self.step())

        random.shuffle(samples_list)
        return samples_list

