import gym
import potion.envs
import numpy as np
from lqg1Dscalable.abstraction.compute_atf.lqg_f_known import LqgFKnown
from lqg1Dscalable.abstraction.compute_atf.lipschitz_f_dads import LipschitzFdads
from lqg1Dscalable.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from lqg1Dscalable.updater_abstract.updater import AbsUpdater
from lqg1Dscalable.updater_abstract.bounded_mdp.IVI import IVI
from lqg1Dscalable.updater_deterministic.updater import Updater
from lqg1Dscalable.visualizer.lqg1d_visualizer import Lqg1dVisualizer
from lqg1Dscalable.abstraction.maxlikelihood_abstraction import MaxLikelihoodAbstraction
import lqg1Dscalable.helper as helper
from lqg1Dscalable.helper import Helper
import logging

problem = 'lqg1d'
SINK = False
INIT_DETERMINISTIC_PARAM = -0.7
ENV_NOISE = 0.1
A = 1
B = 1
GAMMA = 0.9
# optA = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set optA = 0 to use the standard algorithm.
optA = 0
LIPSCHITZ_CONST_STATE = A
LIPSCHITZ_CONST_ACTION = B
LIPSCHITZ_STOCH_ATF = B

# ALFA regulates the update of the deterministic parameter
ALFA = 0.5

N_ITERATION = 60
N_EPISODES = 2000
N_STEPS = 20

# INTERVALS = [[-2, -1.8], [-1.8, -1.6], [-1.6, -1.4], [-1.4, -1.2], [-1.2, -1], [-1, -0.8], [-0.8, -0.6], [-0.6, -0.4],
#              [-0.4, -0.2], [-0.2, -0.1], [-0.1, -0.025], [-0.025, 0.025], [0.025, 0.1], [0.1, 0.2], [0.2, 0.4],
#              [0.4, 0.6], [0.6, 0.8], [0.8, 1], [1, 1.2], [1.2, 1.4], [1.4, 1.6], [1.6, 1.8], [1.8, 2]]

INTERVALS = [[-2, -1.6], [-1.6, -1.2], [-1.2, -0.8], [-0.8, -0.5], [-0.5, -0.3], [-0.3, -0.1], [-0.1, 0.1],
             [0.1, 0.3], [0.3, 0.5], [0.5, 0.8], [0.8, 1.2], [1.2, 1.6], [1.6, 2]]


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


def estimate_performance_abstract_policy(env, n_episodes, n_steps, abstract_policy, init_states):
    acc = 0
    for i in range(0, n_episodes):
        env.reset(init_states[i])
        g = 1
        for j in range(0, n_steps):
            state = env.get_state()
            action = abstract_policy[helper.get_mcrst(state, INTERVALS, SINK)][0]
            new_state, r, _, _ = env.step(action)
            acc += g * r
            g *= GAMMA
    return acc / n_episodes


def main(seed=None):

    help = Helper(seed)

    # load and configure the environment.
    env = gym.make('LQG1D-v0')
    env.sigma_noise = ENV_NOISE
    env.A = np.array([A]).reshape((1, 1))
    env.B = np.array([B]).reshape((1, 1))
    env.gamma = GAMMA
    env.seed(help.getSeed())

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
    det_upd = Updater(help.getSeed())

    title = "A={}, B={}, Opt par={}, Opt J={}, Noise std dev={}".format(A, B, opt_par4vis, optJ4vis, ENV_NOISE)
    key = "{}_{}_{}_{}_{}".format(A, B, ENV_NOISE, det_param, help.getSeed())
    key = key.replace('.', ',')
    key = key + ".jpg"
    initJ = env.computeJ(det_param, 0, N_EPISODES)
    visualizer = Lqg1dVisualizer(title, key, det_param, opt_par4vis, initJ, optJ4vis)
    visualizer.clean_panels()

    # PLOTTER INFO
    stats = {}
    stats['param'] = []
    stats['j'] = []
    stats['sampleJ'] = []
    stats['abstractJ'] = []
    stats['param'].append(det_param)
    stats['j'].append(initJ)
    # ------------

    for i in range(0, N_ITERATION):
        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
        abstraction.divide_samples(determin_samples, problem, help.getSeed())
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

        # ---- performance abstract policy ---
        first_states_ep = [d[0][0] for d in determin_samples]
        absJ = estimate_performance_abstract_policy(env, N_EPISODES, N_STEPS, abs_opt_pol, first_states_ep)
        # ------------------------------------

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, det_param)
        new_det_param = det_upd.batch_gradient_update(det_param, fictitious_samples)
        det_param = ALFA * det_param + (1 - ALFA) * new_det_param

        j = env.computeJ(det_param, 0, N_EPISODES)
        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        print("Updated deterministic policy parameter: {}".format(det_param))
        print("Updated performance measure: {}".format(j))
        print("Updated estimated performance measure: {}\n".format(estj))
        visualizer.show_values(det_param, j, estj, absJ)

        # PLOTTER INFO
        stats['param'].append(det_param)
        stats['j'].append(j)
        stats['sampleJ'].append(estj)
        stats['abstractJ'].append(absJ)
        # ------------

    visualizer.save_image()
    return stats, opt_par4vis, optJ4vis
