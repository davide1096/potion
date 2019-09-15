import gym
import potion.envs
from lqg1D.policy import StochasticPolicy as sp, DeterministicPolicy as dp
from lqg1D import estimator as e
import torch
import random

env = gym.make('LQG1D-v0')
N_SAMPLES = 2000
N_MACROSTATES = 5
N_STEPS = 20
INIT_DETERMINISTIC_PARAM = -0.2

# constants for stochastic policy
INIT_MU = 0.
INIT_OMEGA = 1.
INIT_LR = 0.01

# about macrostates
CONSTANT_INTERVALS = False
INTERVALS = [[-2, -0.4], [-0.4, -0.1], [-0.1, 0.1], [0.1, 0.4], [0.4, 2]]


def sampling_phase():
    det_pol = dp(INIT_DETERMINISTIC_PARAM)
    samples_list = []
    for i in range(0, N_SAMPLES):
        env.reset()

        for j in range(0, N_STEPS):
            state = env.get_state()
            det_pol.train()
            action = det_pol(torch.from_numpy(state).float())
            new_state, r, done, info = env.step(action.detach().numpy())

            # print("[{}] - State = {}, Action = {}, Reward = {}, Next state = {}".format(i, state, action.detach(
            # ).numpy(), r, new_state))
            # store each step I get
            samples_list.append([state[0], action[0], r, new_state[0]])

    random.shuffle(samples_list)
    return samples_list


def get_states_from_samples():
    return [sam[0] for sam in samples]


def updating_phase():
    # estimate the macrostate distribution using the samples I have
    estimate_mcrst_dist = e.estimate_mcrst_dist(get_states_from_samples(), N_MACROSTATES, CONSTANT_INTERVALS,
                                                -env.max_pos, env.max_pos, INTERVALS)
    print(estimate_mcrst_dist)
    for s in samples:
        mcrst = e.get_mcrst_const(s[0], -env.max_pos, env.max_pos, N_MACROSTATES) if CONSTANT_INTERVALS else \
            e.get_mcrst_not_const(s[0], INTERVALS)

        grad_log_pol_mu, grad_log_pol_omega = st_policy[mcrst].gradient_log_policy(s[1])
        # grad = [st.gradient_log_policy(samples[i][j][1]) for st in st_policy]

        # quantities used to perform gradient ascent
        grad_mu = grad_log_pol_mu / estimate_mcrst_dist[mcrst]
        grad_omega = grad_log_pol_omega / estimate_mcrst_dist[mcrst]
        st_policy[mcrst].update_parameters(grad_mu, grad_omega)


def show_results():
    for i in range(0, N_MACROSTATES):
        par = st_policy[i].parameters()
        print("[MCRST{}]".format(i))
        print([p for p in par])


samples = sampling_phase()

# let's calculate a different stochastic policy for every macrostate
st_policy = []
for i in range(0, N_MACROSTATES):
    st_policy.append(sp(INIT_MU, INIT_OMEGA, INIT_LR))

updating_phase()
show_results()
