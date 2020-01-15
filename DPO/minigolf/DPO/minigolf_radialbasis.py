import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_abstract.bounded_mdp.IVI import IVI
from DPO.visualizer.minigolf_visualizer import MGVisualizer
import DPO.helper as helper
from DPO.helper import Helper
import logging
from DPO.minigolf.DPO.RBFNet import RBFNet
import csv


problem = 'minigolf'
SINK = True
ENV_NOISE = 0
GAMMA = 0.99
# ds0 = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set ds0 = 0 to use the standard algorithm that computes bounds related to both space and action distances.
ds0 = 1

N_ITERATION = 501
N_EPISODES = 500
N_STEPS = 20

N_MCRST_DYN = 30
MIN_SPACE_VAL = 0
MAX_SPACE_VAL = 20
# INTERVALS = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16], [16, 18], [18, 20]]
# INTERVALS = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
#              [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20]]
# INTERVALS = [[0, 0.67], [0.67, 1.34], [1.34, 2.01], [2.01, 2.68], [2.68, 3.35], [3.35, 4.02], [4.02, 4.69],
#              [4.69, 5.36], [5.36, 6.03], [6.03, 6.7], [6.7, 7.37], [7.37, 8.04], [8.04, 8.71], [8.71, 9.38],
#              [9.38, 10.05], [10.05, 10.72], [10.72, 11.39], [11.39, 12.06], [12.06, 12.73], [12.73, 13.4],
#              [13.4, 14.07], [14.07, 14.74], [14.74, 15.41], [15.41, 16.08], [16.08, 16.75], [16.75, 17.42],
#              [17.42, 18.09], [18.09, 18.76], [18.76, 19.43], [19.43, 20]]
INTERVALS = [[0, 0.5], [0.5, 1], [1, 2], [2, 3], [3, 4.5], [4.5, 6], [6, 8], [8, 10], [10, 13], [13, 16], [16, 20]]

# radial bases configuration
CENTERS = [4, 8, 12, 16]
STD_DEV = 4
INIT_W = [1, 1, 1, 1]


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


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, rbf, interv):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            prev_action = deterministic_action(np.reshape(s[0], (1, 1)), rbf)
            prev_action = prev_action[0]
            if interv is not None:
                mcrst = helper.get_mcrst(s[0], interv, SINK)
            else:
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

    # logging.basicConfig(level=logging.DEBUG, filename='../../test.log', filemode='w', format='%(message)s')
    cumulative_fail = 0

    filename = "../csv/minigolf/DPO/data{}.csv".format(help.getSeed())
    data_file = open(filename, mode='w')
    file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS)
    # abstraction = MaxLikelihoodAbstraction(GAMMA, SINK, INTERVALS, 5.5)
    abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, 0) if ds0 else IVI(GAMMA, SINK, True, INTERVALS)
    # abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, 0)
    rbf = RBFNet(CENTERS, STD_DEV, INIT_W, help.getSeed())
    # rbf = RBFNet([3, 6, 10, 14, 17], [0.1, 0.3, 0.5, 0.7, 1])
    # rbf = RBFNet([3, 6, 10, 14, 17], [0.49, 0.63, 0.79, 0.95, 1.33], help.getSeed())
    visualizer = MGVisualizer("MG visualizer", "/minigolf/DPO/test{}.jpg".format(help.getSeed()))
    visualizer.clean_panels()

    # PLOTTER INFO
    stats = {}
    stats['w1'] = []
    stats['w2'] = []
    stats['w3'] = []
    stats['w4'] = []
    stats['j'] = []
    stats['fail'] = []
    # ------------

    for i in range(0, N_ITERATION):

        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, rbf)
        # dyn_intervals = helper.build_mcrst_from_samples(determin_samples, N_MCRST_DYN, MIN_SPACE_VAL, MAX_SPACE_VAL)
        dyn_intervals = None
        abstraction.divide_samples(determin_samples, problem, help.getSeed(), intervals=dyn_intervals)
        abstraction.compute_abstract_tf(ds0, ENV_NOISE)

        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container(), intervals=dyn_intervals)

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, rbf, dyn_intervals)
        fictitious_samples = helper.flat_listoflists(fictitious_samples)
        X = [f[0] for f in fictitious_samples]
        y = [f[1] for f in fictitious_samples]
        # X = np.reshape([f[0] for f in fictitious_samples], (len(fictitious_samples),))
        # y = np.reshape([f[1] for f in fictitious_samples], (len(fictitious_samples),))
        rbf.fit(X, y)
        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        print("Iteration n.{}".format(i))
        print("W: {}".format(rbf.w))
        print("Updated estimated performance measure: {}".format(estj))
        zeros, hundred, failing_states = helper.minigolf_reward_counter(determin_samples)
        print("Number of zeroes: {} - Number of big penalties: {}".format(zeros, hundred))
        print("Failing states: {}".format(failing_states))
        cumulative_fail += hundred
        print("Cumulative fails: {}\n".format(cumulative_fail))

        # actions = [m.keys() for m in abstraction.get_container()]
        # action_range = [max(a) - min(a) if len(a) > 0 else 0 for a in actions]
        # intervals = dyn_intervals if dyn_intervals is not None else INTERVALS
        # [print("Mcrst = {}, diameter = {}, action range = {}".format(dyn, dyn[1] - dyn[0], ran)) for dyn, ran in
        #     zip(intervals, action_range)]
        # print("\n")

        w = rbf.w
        visualizer.show_values(w, estj, cumulative_fail)
        file_writer.writerow([w[0], w[1], w[2], w[3], cumulative_fail, estj])

        # PLOTTER INFO
        if i % 10 == 0:
            stats['w1'].append(w[0])
            stats['w2'].append(w[1])
            stats['w3'].append(w[2])
            stats['w4'].append(w[3])
            stats['j'].append(estj)
            stats['fail'].append(cumulative_fail)
        # ------------

    visualizer.save_image()
    return stats
