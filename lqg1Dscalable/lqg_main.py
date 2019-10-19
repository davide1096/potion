import gym
import potion.envs
import random
import numpy as np
from lqg1Dscalable.abstraction.lipschitz_f import LipschitzF
from lqg1Dscalable.abstraction.f_known import FKnown
from lqg1Dscalable.abstraction.lipschitz_deltas import LipschitzDeltaS
from lqg1Dscalable.updater_abstract.updater import AbsUpdater
import lqg1Dscalable.updater_deterministic.updater as det_upd
from lqg1Dscalable.visualizer.lqg1d_visualizer import Lqg1dVisualizer
import lqg1Dscalable.helper as helper

problem = 'lqg1d'
INIT_DETERMINISTIC_PARAM = -0.9
ENV_NOISE = 0
A = 1
B = 1
GAMMA = 0.9
LIPSCHITZ_CONST_F = B

N_ITERATION = 30
N_EPISODES = 2000
N_STEPS = 20

INTERVALS = [[-2, -1.6], [-1.6, -1.2], [-1.2, -1], [-1, -0.8], [-0.8, -0.6], [-0.6, -0.5], [-0.5, -0.4], [-0.4, -0.3],
             [-0.3, -0.2], [-0.2, -0.1], [-0.1, 0.], [0., 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5],
             [0.5, 0.6], [0.6, 0.8], [0.8, 1], [1, 1.2], [1.2, 1.6], [1.6, 2]]


# load and configure the environment.
env = gym.make('LQG1D-v0')
env.sigma_noise = ENV_NOISE
env.A = np.array([A]).reshape((1, 1))
env.B = np.array([B]).reshape((1, 1))
env.seed(helper.SEED)

# calculate the optimal values of the problem.
opt_par4vis = round(env.computeOptimalK()[0][0], 3)
det_param = INIT_DETERMINISTIC_PARAM
optJ4vis = round(env.computeJ(env.computeOptimalK(), ENV_NOISE, N_EPISODES), 3)

# instantiate the components of the algorithm.
# abstraction = LipschitzF(LIPSCHITZ_CONST_F, GAMMA, INTERVALS)
# abstraction = FKnown(A, B, GAMMA, INTERVALS)
abstraction = LipschitzDeltaS(0, B, GAMMA, INTERVALS)
abs_updater = AbsUpdater(GAMMA, INTERVALS)

title = "A={}, B={}, Opt par={}, Opt J={}, Noise std dev={}".format(A, B, opt_par4vis, optJ4vis, ENV_NOISE)
key = "{}_{}_{}_{}".format(A, B, ENV_NOISE, det_param)
key = key.replace('.', ',')
key = key + ".jpg"
initJ = env.computeJ(det_param, ENV_NOISE, N_EPISODES)
visualizer = Lqg1dVisualizer(title, "number of iterations", "parameter", " performance", key, det_param, opt_par4vis,
                             initJ, optJ4vis)


def deterministic_action(det_par, state):
    return det_par * state


def sampling_from_det_pol(env, n_episodes, n_steps, det_par):
    samples_list = []
    for i in range(0, n_episodes):
        env.reset()
        single_sample = []
        for j in range(0, n_steps):
            state = env.get_state()
            action = deterministic_action(det_par, state)
            new_state, r, _, _ = env.step(action)
            single_sample.append([state[0], action[0], r, new_state[0]])
        samples_list.append(single_sample)
    return samples_list


# return for every state the action closest to the previous one among the actions in abs_opt_policy
def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, param):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            prev_action = deterministic_action(param, s[0])
            mcrst = helper.get_mcrst(s[0], INTERVALS)
            if prev_action in abs_opt_policy[mcrst]:
                single_sample.append([s[0], prev_action])
            else:
                index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                single_sample.append([s[0], abs_opt_policy[mcrst][index]])
        fictitious_samples.append(single_sample)
    return fictitious_samples


for i in range(0, N_ITERATION):
    deterministic_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
    abstraction.divide_samples(deterministic_samples, problem)
    abstraction.compute_abstract_tf()
    abstract_optimal_policy = abs_updater.solve_mdp(abstraction.get_container())

    fictitious_samples = sampling_abstract_optimal_pol(abstract_optimal_policy, deterministic_samples, det_param)
    det_param = det_upd.batch_gradient_update(det_param, fictitious_samples)
    j = env.computeJ(det_param, ENV_NOISE, N_EPISODES)
    estj = helper.estimate_J_lqg(deterministic_samples, GAMMA)

    print("Updated deterministic policy parameter: {}".format(det_param))
    print("Updated performance measure: {}".format(j))
    print("Updated abstract performance measure: {}\n".format(estj))
    visualizer.show_values(det_param, j, estj)

visualizer.save_image()

