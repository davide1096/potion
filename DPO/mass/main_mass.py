import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_abstract.bounded_mdp.IVI import IVI
from DPO.algorithm.updater_deterministic.updater import Updater
from DPO.visualizer.lqg1d_visualizer import Lqg1dVisualizer
import DPO.helper as helper
from DPO.helper import Helper
import logging

problem = 'mass'
SINK = False
INIT_DETERMINISTIC_PARAM = np.array([-0.5, -0.1])
ENV_NOISE = 0
TAO = 0.1
MASS = 0.1
A = np.array([[1., TAO], [0., 1.]])
B = np.array([[0.], [TAO / MASS]])
GAMMA = 0.95

# ds0 = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set ds0 = 0 to use the standard algorithm that computes bounds related to both space and action distances.
ds0 = 0

N_ITERATION = 1000
N_EPISODES = 500
N_STEPS = 20

INTERVALS = np.array([[[-1, -0.8], [-0.8, -0.6], [-0.6, -0.4], [-0.4, -0.25], [-0.25, -0.1], [-0.1, 0.1], [0.1, 0.25],
             [0.25, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1]],
             [[-2, -1.6], [-1.6, -1.2], [-1.2, -0.8], [-0.8, -0.5], [-0.5, -0.25], [-0.25, -0.1], [-0.1, 0.1],
              [0.1, 0.25], [0.25, 0.5], [0.5, 0.8], [0.8, 1.2], [1.2, 1.6], [1.6, 2]]])

N_MCRST_DYN = np.array([11, 13])
MIN_SPACE_VAL = np.array([-1, -2])
MAX_SPACE_VAL = np.array([1, 2])


def deterministic_action(det_par, state):
    return np.dot(det_par, state)


def sampling_from_det_pol(env, n_episodes, n_steps, det_par):
    samples_list = []
    for i in range(0, n_episodes):
        env.reset()
        single_sample = []
        for j in range(0, n_steps):
            state = env.get_state()
            action = deterministic_action(det_par, state)[0]
            new_state, r, _, _ = env.step(action)
            single_sample.append([state, action, r, new_state])
        samples_list.append(single_sample)
    return samples_list


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, param, interv):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            prev_action = deterministic_action(param, s[0])
            if interv is not None:
                mcrst_provv = helper.get_mcrst(s[0], interv, SINK)
                mcrst = helper.get_multidim_mcrst(mcrst_provv, interv)
            else:
                mcrst_provv = helper.get_mcrst(s[0], INTERVALS, SINK)
                mcrst = helper.get_multidim_mcrst(mcrst_provv, INTERVALS)
            if prev_action in abs_opt_policy[mcrst]:
                single_sample.append([s[0], prev_action])
            else:
                index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                single_sample.append([s[0], abs_opt_policy[mcrst][index]])
        fictitious_samples.append(single_sample)
    return fictitious_samples


def estimate_performance_abstract_policy(env, n_episodes, n_steps, abstract_policy, init_states, interv):
    acc = 0
    for i in range(0, n_episodes):
        env.reset(init_states[i])
        g = 1
        for j in range(0, n_steps):
            state = env.get_state()
            if interv is not None:
                mcrst = helper.get_mcrst(state, interv, SINK)
                action = abstract_policy[helper.get_multidim_mcrst(mcrst, interv)][0]
            else:
                mcrst = helper.get_mcrst(state, INTERVALS, SINK)
                action = abstract_policy[helper.get_multidim_mcrst(mcrst, INTERVALS)][0]
            new_state, r, _, _ = env.step(action)
            acc += g * r
            g *= GAMMA
    return acc / n_episodes


def main(seed=None):

    help = Helper(seed)

    # load and configure the environment.
    env = gym.make('mass-v0')
    env.sigma_noise = ENV_NOISE
    env.A = A
    env.B = B
    env.gamma = GAMMA
    env.seed(help.getSeed())

    # calculate the optimal values of the problem.
    opt_par = env.computeOptimalK()
    det_param = INIT_DETERMINISTIC_PARAM.reshape(opt_par.shape)
    optJ4vis = round(env.computeJ(env.computeOptimalK(), 0, N_EPISODES), 3)
    logging.basicConfig(level=logging.DEBUG, filename='../test.log', filemode='w', format='%(message)s')

    # instantiate the components of the algorithm.
    lip_s_deltas = A - np.eye(det_param.size)
    lip_a_deltas = B
    abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS, lip_s_deltas, lip_a_deltas, env.Q, env.R)
    # abstraction = MaxLikelihoodAbstraction(GAMMA, SINK, INTERVALS, B)

    abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS) if ds0 else IVI(GAMMA, SINK, True, INTERVALS)
    # abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)
    det_upd = Updater(help.getSeed())

    opt_par4vis = np.round(opt_par, 3)
    # title = "A={}, B={}, Opt par={}, Opt J={}, Noise std dev={}".format(A.item(), B.item(), opt_par4vis,
    #                                                                     optJ4vis, ENV_NOISE)
    # key = "{}_{}_{}_{}_{}".format(A.item(), B.item(), ENV_NOISE, det_param.item(), help.getSeed())
    # key = key.replace('.', ',')
    # key = key + ".jpg"
    initJ = env.computeJ(det_param, 0, N_EPISODES)
    # visualizer = Lqg1dVisualizer(title, key, det_param, opt_par4vis, initJ, optJ4vis)
    # visualizer.clean_panels()

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
        dyn_intervals = helper.build_mcrst_from_samples(determin_samples, N_MCRST_DYN, MIN_SPACE_VAL, MAX_SPACE_VAL)
        # dyn_intervals = None
        abstraction.divide_samples(determin_samples, problem, help.getSeed(), intervals=dyn_intervals)
        abstraction.compute_abstract_tf(ds0, ENV_NOISE)
        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container(), intervals=dyn_intervals)

        # ---- performance abstract policy ---
        first_states_ep = [d[0][0] for d in determin_samples]
        absJ = estimate_performance_abstract_policy(env, N_EPISODES, N_STEPS, abs_opt_pol, first_states_ep,
                                                    dyn_intervals)
        # ------------------------------------

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, det_param, dyn_intervals)
        det_param = det_upd.batch_gradient_update(det_param, fictitious_samples)

        j = env.computeJ(det_param, 0, N_EPISODES)
        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        print("Updated deterministic policy parameter: {}".format(det_param))
        print("Updated performance measure: {}".format(j))
        print("Updated estimated performance measure: {}".format(estj))
        print("Updated estimated abstract performance measure: {}\n".format(absJ))
        # visualizer.show_values(det_param.item(), j, estj, absJ)

        # PLOTTER INFO
        stats['param'].append(det_param)
        stats['j'].append(j)
        stats['sampleJ'].append(estj)
        stats['abstractJ'].append(absJ)
        # ------------

    # visualizer.save_image()
    return stats, opt_par4vis, optJ4vis


main(0)
