import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_deterministic.updater import Updater
import DPO.safetygym.DPO.base_env as base_env
import DPO.helper as helper
from DPO.helper import Helper
import sys
import csv
import os

problem = 'safety'
SINK = False
ACCEPTED_STATES = [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]

# ds0 = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set ds0 = 0 to use the standard algorithm that computes bounds related to both space and action distances.
ds0 = 1

N_ITERATION = 100
N_EPISODES = 10
N_STEPS = 200

offset = 20


def deterministic_action(det_par, state):
    return np.dot(det_par, state)


def compute_state_bounds(samples):
    samples = helper.flat_listoflists(samples)
    min_values = np.minimum(samples[0][0], samples[0][3])
    max_values = np.maximum(samples[0][0], samples[0][3])
    for sam in samples[1:]:
        min_values = np.minimum(min_values, sam[3])
        max_values = np.maximum(max_values, sam[3])
    return min_values, max_values


def manage_observation_state(obs):
    obs = np.array([o for o, i in zip(obs, ACCEPTED_STATES) if i])
    obs[2], obs[3] = helper.bias_compass_observation(obs[2], obs[3], offset)
    return obs


def sampling_from_det_pol(env, n_episodes, n_steps, det_par):
    samples_list = []
    for _ in range(n_episodes):
        obs = env.reset()
        obs = manage_observation_state(obs)
        single_sample = []
        for _ in range(n_steps):
            action = deterministic_action(det_par, obs)
            new_obs, r, _, _ = env.step(action)
            new_obs = manage_observation_state(new_obs)
            single_sample.append([obs, action, r, new_obs])
            obs = new_obs
        samples_list.append(single_sample)
    return samples_list


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, param, interv):
    fictitious_samples = []
    for s in det_samples:
        prev_action = deterministic_action(param, s[0])
        mcrst_provv = helper.get_mcrst(s[0], interv, SINK)
        index_mcrst = helper.get_index_from_mcrst(mcrst_provv, interv)
        if index_mcrst in abs_opt_policy.keys():
            if helper.array_in(prev_action, abs_opt_policy[index_mcrst]):
                fictitious_samples.append([s[0], prev_action])
            else:
                index = np.argmin([helper.sq_distance(act, prev_action) for act in abs_opt_policy[index_mcrst]])
                fictitious_samples.append([s[0], abs_opt_policy[index_mcrst][index]])
    return fictitious_samples


def main(seed=42, lam=0.05):

    help = Helper(seed)
    GAMMA = 1

    # load and configure the environment.
    env = base_env.create_env(seed)

    state_dim = 9
    action_dim = 2
    N_MCRST_DYN = np.full((state_dim, ), 5)
    # N_MCRST_DYN = np.array([5, 5, 4, 4, 5, 2, 2, 3, 3])

    # INIT_DETERMINISTIC_PARAM = np.array([np.full((state_dim, ), 0.1), np.full((state_dim, ), 0.2)])

    # policy learnt with pgpe.
    p = np.array([0.10787985473871231, 0.02179303579032421, 4.300711154937744, 0.10839951038360596,
                  0.017089104279875755, 0.1119314506649971, 0.018646063283085823, -0.17877089977264404,
                  -0.03759196400642395, -0.004248579498380423, 0.48613205552101135, 0.10498402267694473,
                  -12.068914413452148, 1.0702580213546753, -0.04661020636558533, -0.22232159972190857,
                  0.0361342579126358, -0.39843615889549255])

    INIT_DETERMINISTIC_PARAM = p.reshape((action_dim, state_dim))
    det_param = INIT_DETERMINISTIC_PARAM

    filename = "../../csv/safetygym/LAM={}/data{}.csv".format(lam, help.getSeed())
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data_file = open(filename, mode='w')
    file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['j'])

    filename2 = "../../csv/safetygym/LAM={}/appB{}.csv".format(lam, help.getSeed())
    os.makedirs(os.path.dirname(filename2), exist_ok=True)
    data_file2 = open(filename2, mode='w')
    file_writer2 = csv.writer(data_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer2.writerow(['mcrst0', 'mcrst1', 'it0', 'it100'])

    for i in range(0, N_ITERATION):
        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
        samples = helper.flat_listoflists(determin_samples)

        # instantiate the components of the algorithm.
        if i == 0:
            MIN_SPACE_VAL, MAX_SPACE_VAL = compute_state_bounds(determin_samples)
            INTERVALS = helper.get_constant_intervals(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN)
            print("INTERVALS: {}\n{}\n".format(N_MCRST_DYN, INTERVALS))

            abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS)
            abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)
            det_upd = Updater(help.getSeed())

            # ---- compute info for appendix E (mcrst population) ----
            mask = [0, 0, 1, 1, 0, 0, 0, 0, 0]
            res_0 = helper.appendix_mcrst_population(samples, mask, INTERVALS)
            # --------------------------------------------------------

        abstraction.divide_samples(samples, problem, help.getSeed())
        abstraction.compute_abstract_tf()
        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())

        # ---- compute info for appendix E (mcrst population) ----
        if i == 99:
            res_100 = helper.appendix_mcrst_population(samples, mask, INTERVALS)
            for j in range(len(res_0)):
                for k in range(len(res_0[j])):
                    file_writer2.writerow([j, k, res_0[j][k], res_100[j][k]])
            data_file2.close()
        # --------------------------------------------------------

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, samples, det_param, INTERVALS)
        det_param = det_upd.gradient_update(det_param, fictitious_samples, lam)

        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        file_writer.writerow([estj])

        pol = "["
        for d_r in det_param:
            for d in d_r:
                pol = pol+'{},'.format(d)
        pol = pol + ']'

        print("{} - Updated deterministic policy parameter: {}".format(i, pol))
        print("Updated estimated performance measure: {}\n".format(estj))

    data_file.close()

    
# if __name__ == "__main__":
#     main(int(sys.argv[1]))
main(0)
