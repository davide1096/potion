from gym.utils import seeding
import lqg1D.lqgspo as lqgspo
import numpy as np
import scipy.stats as stats

SEED = None
INIT_V = 0.1
# learning rate used to update with policy gradient the abstract policy
LR_POLICY = 10
LR_VFUN = 0.1


# since we have the parameters related to the abstract transition functions we proceed in this way:
# 1) calculate p(x'|x, a) for all x'
# 2) divide the space [0,1) according to these probabilities
# 3) select the macrostate related to the space that contains the rdm_number
def calc_new_state(w_x, b, ac, rdm_number):
    if rdm_number == 0:
        return 0
    den = np.sum(np.exp(w_x * ac + b[0]))
    prob = np.exp(w_x * ac + b[0]) / den
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

    def __init__(self, functions, min_action, max_action, n_samples, n_steps):
        super().__init__()
        # in sel.functions are contained abstract policy, abstract transition and reward functions
        self.functions = functions
        self.np_random, seed = seeding.np_random(SEED)
        self.mcrst_intervals = functions.get_mcrst_intervals()
        self.state = None
        self.reset()
        self.min_action = min_action
        self.max_action = max_action
        self.v_params = np.array([INIT_V for i in range(0, self.functions.n_mcrst)])
        self.n_samples = n_samples
        self.n_steps = n_steps

    def step(self):
        a = self.draw_action_gaussian_policy(self.state) if not self.functions.abstract_policy_version else \
            self.draw_action_weighted_policy(self.state)
        r = lqgspo.abstract_reward_function(self.mcrst_intervals[self.state], a)
        w_params, b_params = self.functions.get_tf_parameters(self.state)
        new_s = calc_new_state(w_params.detach().numpy(), b_params.detach().numpy(), a, self.np_random.uniform(0, 1))
        state = self.state
        self.state = new_s
        return [state, a, r, new_s]

    def reset(self):
        self.state = int(self.np_random.uniform(low=0, high=len(self.mcrst_intervals)))

    def sampling(self):
        samples_list = []
        for i in range(0, self.n_samples):
            self.reset()
            for j in range(0, self.n_steps):
                samples_list.append(self.step())
        # random.shuffle(samples_list)
        return samples_list

    def draw_action_gaussian_policy(self, state):
        # draw an action according to the stochastic policy defined for the current macrostate
        par = self.functions.get_policy_parameters(state)
        mu = next(par).detach()
        sigma = np.exp(next(par).detach())
        a = stats.truncnorm((self.min_action - mu) / sigma, (self.max_action - mu) / sigma, loc=mu, scale=sigma).rvs()
        return a

    # draw an action according to the stochastic policy defined weighting every action performed in the previous samples
    def draw_action_weighted_policy(self, state):
        policy = self.functions.stoch_policy[state]
        # rdm_number between 1 and the #samples started in that macrostate
        rdm_number = self.np_random.random() * sum(policy.values())
        accumulator = 0
        for k in policy.keys():
            accumulator += policy[k]
            if accumulator >= rdm_number:
                # k is a key -> an action
                return float(k)

    def policy_gradient_update(self, samples):
        # I store the rewards array because I want to normalize them
        rewards = np.array([sam[2] for sam in samples])
        d_factor = 1
        index = 0
        # to avoid that actions with p=1 influence the future sampling
        if self.functions.abstract_policy_version:
            self.functions.reset_stoch_policy()
            # todo
            for s in samples:
                if '{}'.format(s[1]) in self.functions.stoch_policy[s[0]]:
                    self.functions.stoch_policy[s[0]]['{:.3}'.format(s[1])] += 1
                else:
                    self.functions.stoch_policy[s[0]]['{:.3}'.format(s[1])] = 1

        for s in samples:
            r_norm = (s[2] - rewards.mean()) / (rewards.std() + 1e-9)
            delta = r_norm + self.functions.gamma * self.v_params[s[3]] - self.v_params[s[0]]
            # update v_params
            self.v_params[s[0]] += LR_VFUN * delta

            # update abstract policy parameters
            if not self.functions.abstract_policy_version:
                grad_log_pol_mu, grad_log_pol_omega = self.functions.stoch_policy[s[0]].gradient_log_policy(s[1])
                upd_mu = d_factor * delta * grad_log_pol_mu
                upd_omega = d_factor * delta * grad_log_pol_omega
                self.functions.stoch_policy[s[0]].update_parameters(upd_mu, upd_omega, LR_POLICY)
            else:
                # tot = sum(self.functions.stoch_policy[s[0]].values())
                prob = self.functions.stoch_policy[s[0]]['{}'.format(s[1])]
                self.functions.stoch_policy[s[0]]['{}'.format(s[1])] += LR_POLICY * d_factor * delta * (1 / prob)
                # if '{}'.format(s[1]) in self.functions.stoch_policy[s[0]]:
                #     self.functions.stoch_policy[s[0]]['{}'.format(s[1])] += + LR_POLICY * d_factor * delta
                # else:
                #     self.functions.stoch_policy[s[0]]['{}'.format(s[1])] = 1 + LR_POLICY * d_factor * delta
                # to avoid p<0
                if self.functions.stoch_policy[s[0]]['{}'.format(s[1])] < 0:
                    self.functions.stoch_policy[s[0]]['{}'.format(s[1])] = 0.001

            # during each episode the discount factor needs to be updated
            d_factor = d_factor * self.functions.gamma if index < (self.n_steps - 1) else 1
            index = index + 1 if index < (self.n_steps - 1) else 0

        # to avoid p<0
        # for sp in self.functions.stoch_policy:
        #     min_val = min(sp.values())
        #     if min_val < 0:
        #         for k in sp.keys():
        #             sp[k] -= min_val



    def show_critic_vparams(self):
        print("V parameters: {}".format(self.v_params))



