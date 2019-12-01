import gym
import potion.envs
import numpy as np
from lqg1Dscalable.abstraction.compute_atf.lqg_f_known import LqgFKnown
from lqg1Dscalable.abstraction.compute_atf.lipschitz_f_dads import LipschitzFdads
from lqg1Dscalable.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from lqg1Dscalable.updater_abstract.updater import AbsUpdater
from lqg1Dscalable.updater_abstract.bounded_mdp.IVI import IVI
import lqg1Dscalable.updater_deterministic.updater as det_upd
from lqg1Dscalable.visualizer.lqg1d_visualizer import Lqg1dVisualizer
from lqg1Dscalable.abstraction.maxlikelihood_abstraction import MaxLikelihoodAbstraction
import lqg1Dscalable.helper as helper
import logging

problem = 'lqg1d'
SINK = False
INIT_DETERMINISTIC_PARAM = -0.4
ENV_NOISE = 0.1
A = 0.7
B = 1
GAMMA = 0.9
# optA = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set optA = 0 to use the standard algorithm.
optA = 0
LIPSCHITZ_CONST_STATE = A
LIPSCHITZ_CONST_ACTION = B
LIPSCHITZ_STOCH_ATF = B

N_ITERATION = 500
N_EPISODES = 2000
N_STEPS = 20

# INTERVALS = [[-2, -1.8], [-1.8, -1.6], [-1.6, -1.4], [-1.4, -1.2], [-1.2, -1], [-1, -0.8], [-0.8, -0.6], [-0.6, -0.4],
#              [-0.4, -0.2], [-0.2, -0.1], [-0.1, 0], [0, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1],
#              [1, 1.2], [1.2, 1.4], [1.4, 1.6], [1.6, 1.8], [1.8, 2]]

INTERVALS = [[-2, -1.6], [-1.6, -1.2], [-1.2, -0.8], [-0.8, -0.5], [-0.5, -0.3], [-0.3, -0.1], [-0.1, 0.1],
             [0.1, 0.3], [0.3, 0.5], [0.5, 0.8], [0.8, 1.2], [1.2, 1.6], [1.6, 2]]


# load and configure the environment.
env = gym.make('LQG1D-v0')
env.sigma_noise = ENV_NOISE
env.A = np.array([A]).reshape((1, 1))
env.B = np.array([B]).reshape((1, 1))
env.gamma = GAMMA
env.seed(helper.SEED)

# calculate the optimal values of the problem.
opt_par4vis = round(env.computeOptimalK()[0][0], 3)
det_param = INIT_DETERMINISTIC_PARAM
optJ4vis = round(env.computeJ(env.computeOptimalK(), 0, N_EPISODES), 3)
logging.basicConfig(level=logging.DEBUG, filename='test.log', filemode='w', format='%(message)s')

# instantiate the components of the algorithm.
# abstraction = LipschitzFdads(LIPSCHITZ_CONST_STATE, LIPSCHITZ_CONST_ACTION, GAMMA, SINK, A, B, INTERVALS)
# abstraction = LqgFKnown(A, B, GAMMA, SINK, INTERVALS)
# abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS, A, B)
abstraction = MaxLikelihoodAbstraction(GAMMA, SINK, INTERVALS, B)

# abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS) if optA else IVI(GAMMA, SINK, True, INTERVALS)
abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)

title = "A={}, B={}, Opt par={}, Opt J={}, Noise std dev={}".format(A, B, opt_par4vis, optJ4vis, ENV_NOISE)
key = "{}_{}_{}_{}".format(A, B, ENV_NOISE, det_param)
key = key.replace('.', ',')
key = key + ".jpg"
initJ = env.computeJ(det_param, 0, N_EPISODES)
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


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, param):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            prev_action = deterministic_action(param, s[0])
            mcrst = helper.get_mcrst(s[0], INTERVALS, SINK)
            if prev_action in abs_opt_policy[mcrst]:
                single_sample.append([s[0], prev_action])
            else:
                index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                single_sample.append([s[0], abs_opt_policy[mcrst][index]])
        fictitious_samples.append(single_sample)
    return fictitious_samples


for i in range(0, N_ITERATION):
    determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
    abstraction.divide_samples(determin_samples, problem)
    abstraction.compute_abstract_tf(optA, ENV_NOISE)

    # --- LOG ---
    # min_action = [min(list(cont.keys())) for cont in abstraction.get_container()]
    # max_action = [max(list(cont.keys())) for cont in abstraction.get_container()]
    # logging.debug("Parameter: {}\n".format(det_param))
    #
    # for i in range(0, len(INTERVALS)):
    #     logging.debug("Macrostate {} - min action: {}".format(i, min_action[i]))
    #     logging.debug(abstraction.get_container()[i][min_action[i]]['abs_tf'])
    #     logging.debug("\n")
    #     logging.debug("Macrostate {} - max action: {}".format(i, max_action[i]))
    #     logging.debug(abstraction.get_container()[i][max_action[i]]['abs_tf'])
    #     logging.debug("\n")
    # -----------

    abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())
    # logging.debug([min(a) for a in abstract_optimal_policy])
    # logging.debug("\n")
    logging.debug("Optimal policy: {}".format(abs_opt_pol))

    fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, det_param)
    det_param = det_upd.batch_gradient_update(det_param, fictitious_samples)
    j = env.computeJ(det_param, 0, N_EPISODES)
    estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

    print("Updated deterministic policy parameter: {}".format(det_param))
    print("Updated performance measure: {}".format(j))
    print("Updated estimated performance measure: {}\n".format(estj))
    visualizer.show_values(det_param, j, estj)

visualizer.save_image()

