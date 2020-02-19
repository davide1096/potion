import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_deterministic.updater import Updater
import DPO.safetygym.base_env as base_env
import DPO.helper as helper
from DPO.helper import Helper
import csv
import os

problem = 'safety'
SINK = False
ACCEPTED_STATES = [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]

state_dim = 9
action_dim = 2
offset = 20


def deterministic_action(det_par, state):
    return np.dot(det_par, state)


def compute_state_bounds(samples):
    # initialize min and max values with the first sample.
    min_values = np.minimum(samples[0][0], samples[0][3])
    max_values = np.maximum(samples[0][0], samples[0][3])
    # evaluate min max throughout all the samples.
    for sam in samples[1:]:
        min_values = np.minimum(min_values, sam[3])
        max_values = np.maximum(max_values, sam[3])
    return min_values, max_values


def manage_observation_state(obs):
    # obs space reduced to the meaningful dimensions.
    obs = np.array([o for o, i in zip(obs, ACCEPTED_STATES) if i])
    # simulated error introduced in the compass observations.
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
        # prev_action is the action that would be prescribed by the (not yet updated) deterministic policy.
        prev_action = deterministic_action(param, s[0])
        mcrst_provv = helper.get_mcrst(s[0], interv, SINK)
        index_mcrst = helper.get_index_from_mcrst(mcrst_provv, interv)
        if index_mcrst in abs_opt_policy.keys():
            if helper.array_in(prev_action, abs_opt_policy[index_mcrst]):  # prev_action is optimal.
                fictitious_samples.append([s[0], prev_action])
            else:  # we select the closest action to prev_action among the optimal ones.
                index = np.argmin([helper.arr_distance(act, prev_action) for act in abs_opt_policy[index_mcrst]])
                fictitious_samples.append([s[0], abs_opt_policy[index_mcrst][index]])
    return fictitious_samples


def main(seed, args):
    alpha = 0.005 if args['alpha'] is None else args['alpha']
    lam = 0.05 if args['lambda'] is None else args['lambda']
    N_MCRST_DYN = np.full((state_dim, ), 5) if args['mcrst'] is None else np.full((state_dim, ), args['mcrst'])
    N_ITERATION = 100 if args['niter'] is None else args['niter']
    N_EPISODES = 10 if args['batch'] is None else args['batch']
    N_STEPS = 200 if args['nsteps'] is None else args['nsteps']

    help = Helper(seed)
    GAMMA = 1
    env = base_env.create_env(seed)  # load and configure the environment.
    p = np.array([0.10787985473871231, 0.02179303579032421, 4.300711154937744, 0.10839951038360596,
                  0.017089104279875755, 0.1119314506649971, 0.018646063283085823, -0.17877089977264404,
                  -0.03759196400642395, -0.004248579498380423, 0.48613205552101135, 0.10498402267694473,
                  -12.068914413452148, 1.0702580213546753, -0.04661020636558533, -0.22232159972190857,
                  0.0361342579126358, -0.39843615889549255])  # policy learnt with pgpe.
    INIT_DETERMINISTIC_PARAM = p.reshape((action_dim, state_dim))
    det_param = INIT_DETERMINISTIC_PARAM

    filename = "../../csv/safetygym/LAM={}/data{}.csv".format(lam, help.getSeed())
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data_file = open(filename, mode='w')
    file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['j'])

    for i in range(0, N_ITERATION):
        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
        flat_samples = helper.flat_listoflists(determin_samples)

        if i == 0:
            # build the discretization.
            MIN_SPACE_VAL, MAX_SPACE_VAL = compute_state_bounds(flat_samples)
            INTERVALS = helper.get_constant_intervals(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN)
            print("Seed: {} - Alpha: {}, Lambda: {}".format(seed, alpha, lam))
            print("INTERVALS: {}\n{}\n".format(N_MCRST_DYN, INTERVALS))
            # instantiate the components of the algorithm.
            abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS)
            abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)
            det_upd = Updater(help.getSeed())

        # build the \gamma-MDP.
        abstraction.divide_samples(flat_samples, problem, help.getSeed())
        abstraction.compute_abstract_tf()
        # compute the value iteration.
        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())
        # project back the policy.
        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, flat_samples, det_param, INTERVALS)
        det_param = det_upd.gradient_update(det_param, fictitious_samples, alpha, lam)

        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        # show the results of the iteration.
        print("Seed {} - Iteration N.{}".format(seed, i))
        print("Policy parameters: {}".format(det_param))
        print("Estimated performance measure: {}\n".format(estj))
        file_writer.writerow([estj])

    data_file.close()
