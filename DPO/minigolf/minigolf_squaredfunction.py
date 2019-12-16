import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_abstract.bounded_mdp.IVI import IVI
import DPO.algorithm.updater_deterministic.updater as det_upd
import DPO.helper as helper
import logging

problem = 'minigolf'
SINK = True
INIT_DETERMINISTIC_PARAM_A = 0.5
INIT_DETERMINISTIC_PARAM_B = 1
ENV_NOISE = 0
GAMMA = 0.99
# optA = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set optA = 0 to use the standard algorithm.
optA = 1

N_ITERATION = 500
N_EPISODES = 1000
N_STEPS = 200

INTERVALS = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16], [16, 18], [18, 20]]

# load and configure the environment.
env = gym.make('MiniGolf-v0')
env.sigma_noise = ENV_NOISE
env.gamma = GAMMA
env.seed(helper.SEED)

# calculate the optimal values of the problem.
det_param_a = INIT_DETERMINISTIC_PARAM_A
det_param_b = INIT_DETERMINISTIC_PARAM_B
logging.basicConfig(level=logging.DEBUG, filename='../test.log', filemode='w', format='%(message)s')

abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS)
abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, 0) if optA else IVI(GAMMA, SINK, True, INTERVALS)


def deterministic_action(det_par_a, det_param_b, state):
    return np.sqrt(det_par_a * state + det_param_b)


def sampling_from_det_pol(env, n_episodes, n_steps, det_par_a, det_param_b):
    samples_list = []
    for j in range(0, n_episodes):
        env.reset()
        k = 0
        single_sample = []
        done = False
        while k < n_steps and not done:
            state = env.get_state()
            action = deterministic_action(det_par_a, det_param_b, state)
            new_state, r, done, _ = env.step(action)
            single_sample.append([state[0], action[0], r, new_state[0]])
            k += 1

        samples_list.append(single_sample)
    return samples_list


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, param_a, param_b):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            prev_action = deterministic_action(param_a, param_b, s[0])
            mcrst = helper.get_mcrst(s[0], INTERVALS, SINK)
            if prev_action in abs_opt_policy[mcrst]:
                single_sample.append([s[0], prev_action])
            else:
                index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                single_sample.append([s[0], abs_opt_policy[mcrst][index]])
        fictitious_samples.append(single_sample)
    return fictitious_samples


for i in range(0, N_ITERATION):
    determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param_a, det_param_b)
    abstraction.divide_samples(determin_samples, problem)
    abstraction.compute_abstract_tf(optA, ENV_NOISE)

    abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())

    fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, det_param_a, det_param_b)
    det_param_a = det_upd.batch_gradient_update_a(det_param_a, det_param_b, fictitious_samples)
    det_param_b = det_upd.batch_gradient_update_b(det_param_a, det_param_b, fictitious_samples)
    estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

    print("Updated deterministic policy parameter A: {}".format(det_param_a))
    print("Updated deterministic policy parameter B: {}".format(det_param_b))
    print("Updated estimated performance measure: {}\n".format(estj))

