import gym
import potion.envs
import random
import numpy as np
from lqg1Dscalable.abstraction.lipschitz_f import LipschitzF
from lqg1Dscalable.abstraction.f_known import FKnown
from lqg1Dscalable.updater_abstract.updater import AbsUpdater
import lqg1Dscalable.updater_deterministic.updater as det_upd
from lqg1Dscalable.visualizer.lqg1d_visualizer import Lqg1dVisualizer
import lqg1Dscalable.helper as helper

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

# calculate the optimal values of the problem.
opt_par4vis = round(env.computeOptimalK()[0][0], 3)
det_param = INIT_DETERMINISTIC_PARAM
optJ4vis = round(env.computeJ(env.computeOptimalK(), ENV_NOISE, N_EPISODES), 3)

# instantiate the components of the algorithm.
abstraction = LipschitzF(LIPSCHITZ_CONST_F, INTERVALS)
# abstraction = FKnown(A, B, INTERVALS)
abs_updater = AbsUpdater(GAMMA, INTERVALS)

title = "A={}, B={}, Opt par={}, Opt J={}, Variance of noise={}".format(A, B, opt_par4vis, optJ4vis, ENV_NOISE)
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
        for j in range(0, n_steps):
            state = env.get_state()
            action = deterministic_action(det_par, state)
            new_state, r, _, _ = env.step(action)
            samples_list.append([state[0], action[0], r, new_state[0]])
    return samples_list


# return the action closest to the previous one among the optimal actions
def sampling_abstract_optimal_pol(abs_opt_policy, state, param):
    prev_action = deterministic_action(param, state)
    mcrst = helper.get_mcrst(state, INTERVALS)
    if prev_action in abs_opt_policy[mcrst]:
        return prev_action
    index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
    return abs_opt_policy[mcrst][index]


for i in range(0, N_ITERATION):
    deterministic_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
    abstraction.divide_samples(deterministic_samples)
    abstraction.compute_abstract_tf()
    abstract_optimal_policy = abs_updater.solve_mdp(abstraction.get_container())

    fictitious_samples = [[s[0], sampling_abstract_optimal_pol(abstract_optimal_policy,
                                                               s[0], det_param)] for s in deterministic_samples]
    det_param = det_upd.batch_gradient_update(det_param, fictitious_samples)
    j = env.computeJ(det_param, ENV_NOISE, N_EPISODES)

    print("Updated deterministic policy parameter: {}".format(det_param))
    print("Updated performance measure: {}\n".format(j))
    visualizer.show_values(det_param, j)

visualizer.save_image()

