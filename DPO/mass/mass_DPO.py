import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.abstraction.maxlikelihood_abstraction_parallel import MaxLikelihoodAbstraction
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_deterministic.updater import Updater
import DPO.helper as helper
from DPO.helper import Helper
import os
import csv
import sys

problem = 'mass'
SINK = False
INIT_DETERMINISTIC_PARAM = np.array([-0.3, -0.3])
A = np.array([[1., 1.], [0., 1.]])
B = np.array([[0.], [1.]])
Q = np.diag([1., 0.])
R = 0.1 * np.eye(1)
GAMMA = 0.95

# ds0 = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set ds0 = 0 to use the standard algorithm that computes bounds related to both space and action distances.
ds0 = 0

N_ITERATION = 120
N_EPISODES = 500
N_STEPS = 20

ENV_NOISE = 0.1 * np.eye(INIT_DETERMINISTIC_PARAM.size)

N_MCRST_DYN = np.array([9, 9])
MIN_SPACE_VAL = np.array([-1, -1])
MAX_SPACE_VAL = np.array([1, 1])
MAX_ACTION_VAL = 1

STOCH_L_MULTIPLIER = 10


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


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, param, INTERVALS):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            prev_action = deterministic_action(param, s[0])
            mcrst_provv = helper.get_mcrst(s[0], INTERVALS, SINK)
            mcrst = helper.get_multidim_mcrst(mcrst_provv, INTERVALS)
            if abs_opt_policy[mcrst] is not None:
                if prev_action in abs_opt_policy[mcrst]:
                    single_sample.append([s[0], prev_action])
                else:
                    index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                    single_sample.append([s[0], abs_opt_policy[mcrst][index]])
        fictitious_samples.append(single_sample)
    return fictitious_samples


def main(seed=None, alpha=0.025, lam=0.0001):

    help = Helper(seed)

    # load and configure the environment.
    env = gym.make('mass-v0')
    env.sigma_noise = ENV_NOISE
    env.A = A
    env.B = B
    env.Q = Q
    env.R = R
    env.gamma = GAMMA
    env.seed(help.getSeed())

    filename = "../csv/mass/DPO/ALPHA={}/LAM={}/9x9/data{}.csv".format(alpha, lam, help.getSeed())
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data_file = open(filename, mode='w')
    file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['param0', 'param1', 'est j'])

    INTERVALS = helper.get_constant_intervals(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN)
    print("Seed: {} - Alpha: {}, Lambda: {}".format(seed, alpha, lam))
    print("INTERVALS: {}\n{}\n".format(N_MCRST_DYN, INTERVALS))

    det_param = INIT_DETERMINISTIC_PARAM.reshape((1, 2))

    # instantiate the components of the algorithm.
    lip_a_tf = B
    abstraction = MaxLikelihoodAbstraction(GAMMA, SINK, INTERVALS, lip_a_tf * STOCH_L_MULTIPLIER, env.Q, env.R)

    abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)
    det_upd = Updater(help.getSeed(), alpha, lam)
    tot_est_j = 0

    for i in range(0, N_ITERATION):
        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)

        abstraction.divide_samples(determin_samples, problem, help.getSeed())
        abstraction.compute_abstract_tf(ds0, MIN_SPACE_VAL, MAX_SPACE_VAL, MAX_ACTION_VAL, ENV_NOISE)
        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples, det_param, INTERVALS)
        det_param = det_upd.batch_gradient_update(det_param, fictitious_samples)

        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)
        tot_est_j += estj

        print("{} - Updated deterministic policy parameter: {}".format(i, det_param))
        print("Updated estimated performance measure: {}".format(estj))

        file_writer.writerow([det_param[0][0], det_param[0][1], estj])

    data_file.close()


# if __name__ == "__main__":
#     main(int(sys.argv[1]))
main(0)