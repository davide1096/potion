import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_abstract.bounded_mdp.IVI import IVI
from DPO.visualizer.minigolf_visualizer import MGVisualizer, plt_regr_perf, RegrVisualizer, ThreeDVisualizer
import DPO.helper as helper
from DPO.helper import Helper
import logging
from DPO.minigolf.DPO.RBFNet import RBFNet
import csv
import os
import DPO.linear_estimation as ln
from DPO.linear_estimation import Ridge
import errno
import sys
import matplotlib.pyplot as plt

problem = 'minigolf'
SINK = True
ENV_NOISE = 0
GAMMA = 0.99
# ds0 = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set ds0 = 0 to use the standard algorithm that computes bounds related to both space and action distances.
ds0 = 1

N_ITERATION = 200  # 500
N_EPISODES = 500  # 500
N_STEPS = 20

N_MCRST_DYN = 12
MIN_SPACE_VAL = 0
MAX_SPACE_VAL = 20
# INTERVALS = [[0, 0.5], [0.5, 1], [1, 2], [2, 3], [3, 4.5], [4.5, 6], [6, 8], [8, 10], [10, 13], [13, 16], [16, 20]]

# radial bases configuration
CENTERS = [4, 8, 12, 16]
STD_DEV = 4
INIT_W = [1, 1, 1, 1]

# Lipschitz constant on delta s
LDELTAS = 0

# Estimate deltas
EST_DS = 1  # 0 if classic DPO
REG = 0.2  # regularization parameter for ridge regression
L_RATE = 0.0001  # learning rate used for the gradient descent
DECAY = 0  # 1 if learning rate decays with iteration number


# regr_models = [] # Array of regression objects (one for each macrostate, except [-4, 0])

# Plotting parameters
# FIG = 1            # Must be increased by one whenever a visualizer object is initialized

def deterministic_action(state, rbf):
    if state < 0:
        return np.array([0])
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


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, rbf, INTERVALS):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            if s[0] > 0:  # exclude s<0 from fictitious samples
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


def estds(cont, it, regr_models, sink, acc=None):
    # Function that initializes Ridge Regression objects for each macrostate and updates the model each iteration
    l = len(cont) if not sink else len(cont) - 1
    acc_samples = [[] for i in range(3)]
    for i in range(1, l):  # 1 cause first interval is [-4, 0] and has all zeros
        input_actions = np.array(list(cont[i].keys()))
        input_states = np.array([d['state'] for d in cont[i].values()])
        target_states = np.array([d['new_state'] for d in cont[i].values()])
        target_ds = np.subtract(target_states, input_states)
        if it == 0:
            regr_models.append(Ridge(REG, L_RATE, DECAY))
            regr_models[i - 1].ls_fit([input_states, input_actions], target_ds)
        else:
            # Modify this part to change how (and if) the linear models get updated
            # FIRST FIT
            # regr_models[i - 1].compute_error([input_states, input_actions], target_ds)

            # GRAD DESCENT
            # regr_models[i - 1].mb_update([input_states, input_actions], target_ds, it)

            # FIT EACH ITER
            # regr_models[i - 1].compute_error([input_states, input_actions], target_ds)
            # regr_models[i - 1].ls_fit([input_states, input_actions], target_ds)

            # ACCUMULATIVE FIT
            regr_models[i - 1].compute_error([input_states, input_actions], target_ds)
            input_states = np.concatenate((np.array(acc[0][i-1]), input_states))
            input_actions = np.concatenate((np.array(acc[1][i-1]), input_actions))
            target_states = np.concatenate((np.array(acc[2][i-1]), target_states))
            target_ds = np.subtract(target_states, input_states)
            regr_models[i - 1].ls_fit([input_states, input_actions], target_ds)

        acc_samples[0].append(input_states)
        acc_samples[1].append(input_actions)
        acc_samples[2].append(target_states)

    return acc_samples



def main(seed=None, alpha=0.001, lam=0.0005):
    help = Helper(seed)

    # load and configure the environment.
    env = gym.make('ComplexMiniGolf-v0')
    env.sigma_noise = ENV_NOISE
    env.gamma = GAMMA
    env.seed(help.getSeed())

    # logging.basicConfig(level=logging.DEBUG, filename='../../test.log', filemode='w', format='%(message)s')
    cumulative_fail = 0
    cumulative_j = 0

    if not EST_DS:
        csv_dir = "../csv/minigolf/friction1.9/DPO/CLASSIC/ALPHA={}/LAM={}".format(alpha, lam, help.getSeed())
        img_dir = "minigolf/DPO/CLASSIC/ALPHA={}/LAM={}".format(alpha, lam)
    else:
        csv_dir = "../csv/minigolf/friction1.9/DPO/EST_DS/ALPHA={}/LAM={}/REG={}/L_RATE={}/DECAY={}".format(alpha,
                                                                                                            lam,
                                                                                                            REG,
                                                                                                            L_RATE,
                                                                                                            DECAY,
                                                                                                            )
        img_dir = "minigolf/DPO/EST_DS/ALPHA={}/LAM={}/REG={}/L_RATE={}/DECAY={}".format(alpha, lam, REG, L_RATE,
                                                                                         DECAY)

    filename = csv_dir + ("/data{}.csv".format(help.getSeed()))

    # filename = "../csv/minigolf/friction1.9/DPO/CLASSIC/ALPHA={}/LAM={}/data{}.csv".format(alpha, lam,
    #             help.getSeed()) if not EST_DS else "../csv/minigolf/friction1.9/DPO/EST_DS/ALPHA={}/LAM={}/REG={}/L_RATE={}/DECAY={}/data{}.csv".format(alpha, lam, REG, L_RATE,
    #                                                                                     DECAY, help.getSeed())

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data_file = open(filename, mode='w')
    file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    rbf = RBFNet(CENTERS, STD_DEV, INIT_W, help.getSeed(), alpha, lam)
    # rbf = RBFNet([3, 6, 10, 14, 17], [0.1, 0.3, 0.5, 0.7, 1])
    # rbf = RBFNet([3, 6, 10, 14, 17], [0.49, 0.63, 0.79, 0.95, 1.33], help.getSeed())
    # visualizer = MGVisualizer("MG visualizer", "minigolf/DPO/ALPHA={}/LAM={}/test{}.png".format(alpha, lam,
    #                                                                                              help.getSeed()))

    visualizer = MGVisualizer("MG visualizer", img_dir + ("/test_policy{}.png".format(help.getSeed())), 1)
    visualizer.clean_panels()

    r_visualizer = RegrVisualizer("RG visualizer", img_dir + ("/test_regr{}.png".format(help.getSeed())), 2, (15, 12.8))
    r_visualizer.clean_panels()

    deltas_visualizer = ThreeDVisualizer("3D Visualizer", img_dir + ("/deltas{}.png".format(help.getSeed())), 3,
                                         (7 * (N_ITERATION // 50 + 1), 25))

    # PLOTTER INFO
    stats = {}
    stats['w1'] = []
    stats['w2'] = []
    stats['w3'] = []
    stats['w4'] = []
    stats['j'] = []
    stats['fail'] = []
    # ------------

    regr_models = []

    for i in range(0, N_ITERATION):

        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, rbf)

        INTERVALS = helper.get_constant_intervals([MIN_SPACE_VAL], [MAX_SPACE_VAL], [N_MCRST_DYN])[0]
        # dyn_intervals = helper.build_mcrst_from_samples(determin_samples, N_MCRST_DYN, MIN_SPACE_VAL, MAX_SPACE_VAL)
        INTERVALS = [[-4, 0]] + INTERVALS
        dyn_intervals = None

        if i == 0:
            abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS) if ds0 else LipschitzDeltaS(GAMMA, SINK, INTERVALS,
                                                                                              1.3, 0.9)
            # abstraction = MaxLikelihoodAbstraction(GAMMA, SINK, INTERVALS, 5.5)
            abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, -100) if ds0 and LDELTAS == 0 else \
                IVI(GAMMA, SINK, True, INTERVALS)
            # abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, 0)

        abstraction.divide_samples(determin_samples, problem, help.getSeed(), intervals=dyn_intervals)
        cont = abstraction.get_container()
        acc = estds(cont, i, regr_models, SINK, None if i==0 else acc)


        abstraction.compute_abstract_tf(ds0, EST_DS, LDELTAS, models=regr_models if EST_DS else None)

        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container(), intervals=dyn_intervals)

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, rbf, INTERVALS)
        fictitious_samples = helper.flat_listoflists(fictitious_samples)
        X = [f[0] for f in fictitious_samples]
        y = [f[1] for f in fictitious_samples]
        # X = np.reshape([f[0] for f in fictitious_samples], (len(fictitious_samples),))
        # y = np.reshape([f[1] for f in fictitious_samples], (len(fictitious_samples),))
        rbf.fit(X, y)
        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        cumulative_j += estj
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

        # --- APPENDIX E ---
        if i == 0 or i == 99 or i == 199 or i == 299 or i == 399 or i == 499:
            filename2 = "../csv/minigolf/appendix/ALPHA={}/LAM={}/it{}/data{}.csv".format(alpha, lam, i, help.getSeed())
            os.makedirs(os.path.dirname(filename2), exist_ok=True)
            data_file2 = open(filename2, mode='w')
            file_writer2 = csv.writer(data_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer2.writerow(['mcrst', 'min_a', 'max_a', 'min_opt_a', 'max_opt_a', 'w1', 'w2', 'w3', 'w4'])
            for j in range(1, len(abstraction.get_container()) - 1):
                actions = abstraction.get_container()[j].keys()
                w = rbf.w
                file_writer2.writerow([j, min(actions), max(actions), min(abs_opt_pol[j]), max(abs_opt_pol[j]),
                                       w[0], w[1], w[2], w[3]])
            data_file2.close()
        # ------------------

        # PLOTTER INFO
        # if i % 10 == 0:
        stats['w1'].append(w[0])
        stats['w2'].append(w[1])
        stats['w3'].append(w[2])
        stats['w4'].append(w[3])
        stats['j'].append(estj)
        stats['fail'].append(cumulative_fail)
        # ------------

        r_visualizer.show_values([m.get_weights() for m in regr_models], [m.get_error() for m in regr_models])

        deltas_visualizer.show_values([m.get_weights() for m in regr_models], i, N_ITERATION, INTERVALS)

    r_visualizer.save_image()
    visualizer.save_image()
    deltas_visualizer.save_image()
    # plt.subplots_adjust(hspace=0.6, wspace=0.4)visualizer.save_image()
    return stats, cumulative_j

# if __name__ == "__main__":
#     main(int(sys.argv[1]))
# main(0)
