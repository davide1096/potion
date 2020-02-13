import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
import DPO.helper as helper
from DPO.helper import Helper
from DPO.minigolf.RBFNet import RBFNet
import csv
import os


problem = 'minigolf'
SINK = True
ENV_NOISE = 0
GAMMA = 0.99
# ds0 = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set ds0 = 0 to use the standard algorithm that computes bounds related to both space and action distances.
ds0 = 1

N_ITERATION = 500
N_EPISODES = 500
N_STEPS = 20

N_MCRST_DYN = [12]
MIN_SPACE_VAL = [0]
MAX_SPACE_VAL = [20]

# radial bases configuration.
CENTERS = [4, 8, 12, 16]
STD_DEV = 4
INIT_W = [1, 1, 1, 1]

# Lipschitz constant on delta s.
LDELTAS = 0


def deterministic_action(state, rbf):
    if np.all(state[0] < 0):
        return np.array([0])
    return rbf.predict(state)[0]


def sampling_from_det_pol(env, n_episodes, n_steps, rbf):
    samples_list = []
    for _ in range(n_episodes):
        env.reset()
        k = 0
        single_sample = []
        done = False
        while k < n_steps and not done:
            state = env.get_state()
            action = deterministic_action(state, rbf)
            new_state, r, done, _ = env.step(action)
            single_sample.append([state, action, r, new_state])
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
                index_mcrst = helper.get_index_from_mcrst(mcrst, INTERVALS)
                if index_mcrst in abs_opt_policy.keys():
                    if helper.array_in(prev_action, abs_opt_policy[index_mcrst]):  # prev_action is optimal.
                        single_sample.append([s[0], prev_action])
                    else:  # we select the closest action to prev_action among the optimal ones.
                        index = np.argmin([helper.sq_distance(act, prev_action) for act in abs_opt_policy[index_mcrst]])
                        single_sample.append([s[0], abs_opt_policy[index_mcrst][index]])
        fictitious_samples.append(single_sample)
    return fictitious_samples


def main(seed=None, alpha=0.001, lam=0.0005):

    help = Helper(seed)
    env = gym.make('ComplexMiniGolf-v0')  # load and configure the environment.
    env.sigma_noise = ENV_NOISE
    env.gamma = GAMMA
    env.seed(help.getSeed())
    cumulative_fail = 0
    cumulative_j = 0
    rbf = RBFNet(CENTERS, STD_DEV, INIT_W, help.getSeed(), alpha, lam)

    filename = "../csv/minigolf/friction1.9/DPO/ALPHA={}/LAM={}/data{}.csv".format(alpha, lam, help.getSeed())
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data_file = open(filename, mode='w')
    file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for i in range(N_ITERATION):

        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, rbf)
        INTERVALS = helper.get_constant_intervals(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN)
        INTERVALS[0] = [[-4, 0]] + INTERVALS[0]  # add the first macrostate representing the goal.
        flat_samples = helper.flat_listoflists(determin_samples)

        if i == 0:
            # instantiate the components of the algorithm.
            abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS)
            abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, -100)

        abstraction.divide_samples(flat_samples, problem, help.getSeed())
        abstraction.compute_abstract_tf()
        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, rbf, INTERVALS)
        fictitious_samples = helper.flat_listoflists(fictitious_samples)
        X = [f[0] for f in fictitious_samples]
        y = [f[1] for f in fictitious_samples]
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

        w = rbf.w
        file_writer.writerow([w[0], w[1], w[2], w[3], cumulative_fail, estj])

main()