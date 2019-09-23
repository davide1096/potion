import gym
import potion.envs
from lqg1D.policy import StochasticPolicy as sp
from lqg1D import estimator as e
import numpy as np
import lqg1D.transition_function as tf
import torch

# constants for stochastic policy (0 for the Gaussian, 1 for the weighted one obtained from the samples)
ABSTRACT_POLICY_VERSION = 1
INIT_MU = 0.
INIT_OMEGA = 1.
# learning rate used to obtain the Gaussian policy from (deterministic) samples
LR_POLICY = 0.001

# constants for abstract transition function (0 for the 1st definition, 1 for the 2nd one)
TRANSITION_FUNCTION_VERSION = 1
INIT_W = 1.
INIT_B = 0.5
LR_TFUN_W = 0.1
LR_TFUN_B = 0.1

# about macrostates
N_MACROSTATES = 6
CONSTANT_INTERVALS = False
INTERVALS = [[-2, -0.4], [-0.4, -0.1], [-0.1, 0.], [0., 0.1], [0.1, 0.4], [0.4, 2]]
# INTERVALS = [[-2, -1.8], [-1.8, -1.6], [-1.6, -1.4], [-1.4, -1.2], [-1.2, -1], [-1, -0.8], [-0.8, -0.6],
#              [-0.6, -0.4], [-0.4, -0.2], [-0.2, 0], [0, 0.2], [0.2, 0.4], [0.4, 0.6],
#              [0.6, 0.8], [0.8, 1], [1, 1.2], [1.2, 1.4], [1.4, 1.6], [1.6, 1.8], [1.8, 2]]
GAMMA = 0.9


# a fast possibility to implement the abstract reward function
# the offset ensure the reward function to be always positive
def abstract_reward_function(interval, action):
    offset = 4.
    abstract_state = (interval[0] + interval[1]) / 2
    return offset - 0.5 * ((abstract_state * abstract_state) + (action * action))


class LqgSpo(object):

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.n_mcrst = N_MACROSTATES
        self.gamma = GAMMA
        self.abstract_policy_version = ABSTRACT_POLICY_VERSION
        self.estimate_mcrst_dist = None
        self.intervals = INTERVALS

        # let's calculate a different stochastic policy for every macrostate
        self.stoch_policy = None
        self.reset_stoch_policy()

        # in order to represent the abstract transition functions we define a parameter for each pair of macrostates
        tf_wparams = np.full((N_MACROSTATES, N_MACROSTATES), INIT_W)
        # we also need a par that keeps track of the state when the action = 0 is multiplied with the w parameter
        tf_bparams = np.full((1, N_MACROSTATES), INIT_B)
        self.tf_wparams = torch.tensor(tf_wparams, requires_grad=True)
        self.tf_bparams = torch.tensor(tf_bparams, requires_grad=True)

    def reset_stoch_policy(self):
        self.stoch_policy = []
        if not ABSTRACT_POLICY_VERSION:
            for i in range(0, N_MACROSTATES):
                self.stoch_policy.append(sp(INIT_MU, INIT_OMEGA, LR_POLICY, -self.env.max_action, self.env.max_action))
        else:
            for i in range(0, N_MACROSTATES):
                self.stoch_policy.append({})

    def from_states_to_macrostates(self, samples):
        mcrst_samples = [[self.get_mcrst(s[0]), s[1], s[2], self.get_mcrst(s[3])] for s in samples]
        # estimate the macrostate distribution using the samples I have
        state_samples = [sam[0] for sam in mcrst_samples]
        self.estimate_mcrst_dist = e.estimate_mcrst_dist(state_samples, N_MACROSTATES)
        print("Estimate of macrostates distribution: {}".format(self.estimate_mcrst_dist))
        return mcrst_samples

    def update_abs_policy(self, spl):
        self.update_abs_policy_gaussian(spl) if not ABSTRACT_POLICY_VERSION else self.update_abs_policy_weighted(spl)

    def update_abs_policy_gaussian(self, samples):
        for s in samples:
            grad_log_pol_mu, grad_log_pol_omega = self.stoch_policy[s[0]].gradient_log_policy(s[1])
            # quantities used to perform gradient ascent
            grad_mu = grad_log_pol_mu / self.estimate_mcrst_dist[s[0]]
            grad_omega = grad_log_pol_omega / self.estimate_mcrst_dist[s[0]]
            self.stoch_policy[s[0]].update_parameters(grad_mu, grad_omega, LR_POLICY)

    # stores in a dictionary the actions (key) and the number of times they are performed (value)
    def update_abs_policy_weighted(self, samples):
        for s in samples:
            if '{}'.format(s[1]) in self.stoch_policy[s[0]]:
                self.stoch_policy[s[0]]['{:.3}'.format(s[1])] += 1
            else:
                self.stoch_policy[s[0]]['{:.3}'.format(s[1])] = 1
        # TODO to fix it better (huge probability if we have a=0)
        for p in self.stoch_policy:
            if '-0.0' in p:
                del p['-0.0']

    def update_abs_tf(self, samples):
        for s in samples:

            # I obtain all the parameters related to the same starting macrostate
            w_row = self.tf_wparams[s[0]]
            prob = tf.get_grad_tf_prob(w_row, self.tf_bparams, s[3], s[1], TRANSITION_FUNCTION_VERSION)

            if TRANSITION_FUNCTION_VERSION == 0:
                # I calculate the update terms... (negative loss, so I can sum during the update)
                grad_w = (1 - prob) * self.tf_wparams.grad[s[0]] / self.estimate_mcrst_dist[s[0]]
                grad_b = (1 - prob) * self.tf_bparams.grad / self.estimate_mcrst_dist[s[0]]
            else:
                grad_w = self.tf_wparams.grad[s[0]] / self.estimate_mcrst_dist[s[0]]
                grad_b = self.tf_bparams.grad / self.estimate_mcrst_dist[s[0]]

            # update parameters
            with torch.no_grad():
                self.tf_wparams[s[0]] += LR_TFUN_W * grad_w
                self.tf_bparams += LR_TFUN_B * grad_b
            self.tf_wparams.grad.zero_()
            self.tf_bparams.grad.zero_()

    def show_abs_policy_params(self):
        if not ABSTRACT_POLICY_VERSION:
            for i in range(0, N_MACROSTATES):
                par = self.stoch_policy[i].parameters()
                print("[Policy parameters in MCRST{}]".format(i))
                print([p[0] for p in par])

    def get_policy_parameters(self, mcrst):
        return self.stoch_policy[mcrst].parameters() if not ABSTRACT_POLICY_VERSION else None

    def show_abs_tf_params(self):
        print("Parameters of the transition functions between macrostates: ")
        print(self.tf_wparams)
        print(self.tf_bparams)

    # this function calculates the tf probabilities related to the mean action in every macrostate
    def show_tf_prob(self):
        if not ABSTRACT_POLICY_VERSION:
            print("Transition function probabilities between macrostates given the mean action: ")
            for i in range(0, N_MACROSTATES):
                abs_pol_par = self.stoch_policy[i].parameters()
                mu = next(abs_pol_par).detach().numpy()
                prob = [tf.get_tf_prob(self.tf_wparams[i], self.tf_bparams, j, mu)[0] for j in range(0, N_MACROSTATES)]
                print(prob)

    def get_mcrst_intervals(self):
        return e.get_constant_intervals(-self.env.max_pos, self.env.max_pos, N_MACROSTATES) if CONSTANT_INTERVALS \
            else INTERVALS

    def get_mcrst(self, state):
        return e.get_mcrst_const(state, -self.env.max_pos, self.env.max_pos, N_MACROSTATES) if CONSTANT_INTERVALS \
            else e.get_mcrst_not_const(state, INTERVALS)

    def get_tf_parameters(self, mcrst):
        return self.tf_wparams[mcrst], self.tf_bparams
