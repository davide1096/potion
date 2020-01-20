import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_deterministic.updater import Updater
import DPO.safetygym.DPO.base_env as base_env
import DPO.helper as helper
from DPO.helper import Helper

problem = 'safety'
SINK = False
ACCEPTED_STATES = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]

# ds0 = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set ds0 = 0 to use the standard algorithm that computes bounds related to both space and action distances.
ds0 = 1

N_ITERATION = 1000
N_EPISODES = 50
N_STEPS = 20


def deterministic_action(det_par, state):
    return np.dot(det_par, state)


def normalize_samples(samples):
    samples = helper.flat_listoflists(samples)
    min_values = np.minimum(samples[0][0], samples[0][3])
    max_values = np.maximum(samples[0][0], samples[0][3])
    for sam in samples[1:]:
        min_values = np.minimum(min_values, sam[3])
        max_values = np.maximum(max_values, sam[3])
    normalized_samples = []
    den = (max_values - min_values)
    den = np.array([d if d != 0 else 1 for d in den])
    for i in range(len(samples)):
        sam = samples[i]
        norm_s = (sam[0] - min_values) / den
        norm_ns = (sam[3] - min_values) / den
        normalized_samples.append([norm_s, sam[1], sam[2], norm_ns])
    return normalized_samples


def sampling_from_det_pol(env, n_episodes, n_steps, det_par):
    samples_list = []
    for _ in range(n_episodes):
        obs = env.reset()
        obs = np.array([o for o, i in zip(obs, ACCEPTED_STATES) if i])
        single_sample = []
        for _ in range(n_steps):
            action = deterministic_action(det_par, obs)
            new_obs, r, _, _ = env.step(action)
            new_obs = np.array([o for o, i in zip(new_obs, ACCEPTED_STATES) if i])
            single_sample.append([obs, action, r, new_obs])
            obs = new_obs
        samples_list.append(single_sample)
    return samples_list


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, param, interv):
    fictitious_samples = []
    for s in det_samples:
        prev_action = deterministic_action(param, s[0])
        if interv is not None:
            mcrst_provv = helper.get_mcrst(s[0], interv, SINK)
            mcrst = helper.get_multidim_mcrst(mcrst_provv, interv)
        else:
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


def main(seed=42):

    help = Helper(seed)
    GAMMA = 1

    # load and configure the environment.
    env = base_env.create_env(seed)

    state_dim = 7
    action_dim = 2
    MIN_SPACE_VAL = np.full((state_dim, ), 0)
    MAX_SPACE_VAL = np.full((state_dim, ), 1)
    MAX_ACTION_VAL = np.full((action_dim, ), 1)
    N_MCRST_DYN = np.array([2, 2, 2, 2, 2, 2, 2])
    INTERVALS = helper.get_constant_intervals(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN)
    print("INTERVALS: {}\n{}\n".format(N_MCRST_DYN, INTERVALS))

    INIT_DETERMINISTIC_PARAM = np.full((action_dim, state_dim), 0.1)
    det_param = INIT_DETERMINISTIC_PARAM

    # instantiate the components of the algorithm.
    abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS)
    abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)
    det_upd = Updater(help.getSeed())

    for i in range(0, N_ITERATION):
        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
        normalized_samples = normalize_samples(determin_samples)
        abstraction.divide_samples(normalized_samples, problem, help.getSeed())
        abstraction.compute_abstract_tf(ds0, MIN_SPACE_VAL, MAX_SPACE_VAL, MAX_ACTION_VAL)
        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, normalized_samples, det_param, INTERVALS)
        old_par = det_param
        det_param = det_upd.batch_gradient_update(det_param, fictitious_samples)

        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        print("{} - Updated deterministic policy parameter: {}".format(i, det_param))
        print("Updated performance measure: {}".format(j))
        print("Updated estimated performance measure: {}".format(estj))

main(0)
