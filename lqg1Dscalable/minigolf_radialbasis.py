import gym
import potion.envs
import numpy as np
from lqg1Dscalable.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from lqg1Dscalable.updater_abstract.updater import AbsUpdater
from lqg1Dscalable.updater_abstract.bounded_mdp.IVI import IVI
from lqg1Dscalable.abstraction.maxlikelihood_abstraction import MaxLikelihoodAbstraction
from lqg1Dscalable.visualizer.minigolf_visualizer import MGVisualizer
import lqg1Dscalable.helper as helper
from lqg1Dscalable.helper import Helper
import logging
from lqg1Dscalable.RBFNet import RBFNet

problem = 'minigolf'
SINK = True
ENV_NOISE = 0
GAMMA = 0.99
# optA = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set optA = 0 to use the standard algorithm.
optA = 1

N_ITERATION = 401
N_EPISODES = 2000
N_STEPS = 20

INTERVALS = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16], [16, 18], [18, 20]]
# INTERVALS = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
#              [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20]]


def deterministic_action(state, rbf):
    return rbf.predict(state)[0]


def sampling_from_det_pol(env, n_episodes, n_steps, rbf):
    samples_list = []
    for j in range(0, n_episodes):
        env.reset()
        k = 0
        single_sample = []
        done = False
        while k < n_steps and not done:
            state = env.get_state()
            action = deterministic_action(state, rbf)
            new_state, r, done, _ = env.step(action)
            single_sample.append([state[0], action[0], r, new_state[0]])
            k += 1

        samples_list.append(single_sample)
    return samples_list


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, rbf):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            prev_action = deterministic_action(np.reshape(s[0], (1, 1)), rbf)
            prev_action = prev_action[0]
            mcrst = helper.get_mcrst(s[0], INTERVALS, SINK)
            if prev_action in abs_opt_policy[mcrst]:
                single_sample.append([s[0], prev_action])
            else:
                index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                single_sample.append([s[0], abs_opt_policy[mcrst][index]])
        fictitious_samples.append(single_sample)
    return fictitious_samples


def main(seed=None):

    help = Helper(seed)

    # load and configure the environment.
    env = gym.make('MiniGolf-v0')
    env.sigma_noise = ENV_NOISE
    env.gamma = GAMMA
    env.seed(help.getSeed())

    logging.basicConfig(level=logging.DEBUG, filename='test.log', filemode='w', format='%(message)s')

    abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS)
    # abstraction = MaxLikelihoodAbstraction(GAMMA, SINK, INTERVALS, 5.5)
    abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, 0) if optA else IVI(GAMMA, SINK, True, INTERVALS)
    # abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, 0)
    rbf = RBFNet([3, 6, 10, 14, 17], [0.5, 0.5, 0.5, 0.5, 0.5], 1, help.getSeed())
    # rbf = RBFNet([3, 6, 10, 14, 17], [0.1, 0.3, 0.5, 0.7, 1], 1)
    # rbf = RBFNet([3, 6, 10, 14, 17], [0.49, 0.63, 0.79, 0.95, 1.33], 2, help.getSeed())
    visualizer = MGVisualizer("MG visualizer", "test{}.jpg".format(help.getSeed()))
    visualizer.clean_panels()

    # PLOTTER INFO
    stats = {}
    stats['w1'] = []
    stats['w2'] = []
    stats['w3'] = []
    stats['w4'] = []
    stats['w5'] = []
    stats['b'] = []
    stats['j'] = []
    # ------------

    for i in range(0, N_ITERATION):

        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, rbf)
        abstraction.divide_samples(determin_samples, problem, help.getSeed())
        abstraction.compute_abstract_tf(optA, ENV_NOISE)

        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, rbf)
        fictitious_samples = helper.flat_listoflists(fictitious_samples)
        X = np.reshape([f[0] for f in fictitious_samples], (len(fictitious_samples),))
        y = np.reshape([f[1] for f in fictitious_samples], (len(fictitious_samples),))
        rbf.fit(X, y)
        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        print("Iteration n.{}".format(i))
        print("W: {}".format(rbf.w))
        print("b: {}".format(rbf.b))
        print("Updated estimated performance measure: {}".format(estj))
        zeros, hundred, failing_states = helper.minigolf_reward_counter(determin_samples)
        print("Number of zeroes: {} - Number of big penalties: {}".format(zeros, hundred))
        print("Failing states: {}\n".format(failing_states))

        w = rbf.w
        b = rbf.b
        visualizer.show_values(w, b, estj)

        # PLOTTER INFO
        if i % 10 == 0:
            stats['w1'].append(w[0])
            stats['w2'].append(w[1])
            stats['w3'].append(w[2])
            stats['w4'].append(w[3])
            stats['w5'].append(w[4])
            stats['b'].append(b)
            stats['j'].append(estj)
        # ------------

    visualizer.save_image()
    return stats
