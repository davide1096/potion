import gym
import potion.envs
from lqg1D.policy import StochasticPolicy as sp
from lqg1D import estimator as e

# constants for stochastic policy
INIT_MU = 0.
INIT_OMEGA = 1.
INIT_LR = 0.01

# about macrostates
N_MACROSTATES = 5
CONSTANT_INTERVALS = False
INTERVALS = [[-2, -0.4], [-0.4, -0.1], [-0.1, 0.1], [0.1, 0.4], [0.4, 2]]


def get_states_from_samples(samples):
    return [sam[0] for sam in samples]


class LqgSpo(object):

    def __init__(self, env):
        self.env = env
        # let's calculate a different stochastic policy for every macrostate
        self.stoch_policy = []
        for i in range(0, N_MACROSTATES):
            self.stoch_policy.append(sp(INIT_MU, INIT_OMEGA, INIT_LR))

    def update_abs_policy(self, samples):
        # estimate the macrostate distribution using the samples I have
        estimate_mcrst_dist = e.estimate_mcrst_dist(get_states_from_samples(samples), N_MACROSTATES, CONSTANT_INTERVALS,
                                                    -self.env.max_pos, self.env.max_pos, INTERVALS)
        print(estimate_mcrst_dist)
        for s in samples:
            mcrst = e.get_mcrst_const(s[0], -self.env.max_pos, self.env.max_pos, N_MACROSTATES) if CONSTANT_INTERVALS \
                else e.get_mcrst_not_const(s[0], INTERVALS)
            grad_log_pol_mu, grad_log_pol_omega = self.stoch_policy[mcrst].gradient_log_policy(s[1])
            # grad = [st.gradient_log_policy(samples[i][j][1]) for st in st_policy]

            # quantities used to perform gradient ascent
            grad_mu = grad_log_pol_mu / estimate_mcrst_dist[mcrst]
            grad_omega = grad_log_pol_omega / estimate_mcrst_dist[mcrst]
            self.stoch_policy[mcrst].update_parameters(grad_mu, grad_omega)

    def show_abs_policy_params(self):
        for i in range(0, N_MACROSTATES):
            par = self.stoch_policy[i].parameters()
            print("[MCRST{}]".format(i))
            print([p for p in par])

