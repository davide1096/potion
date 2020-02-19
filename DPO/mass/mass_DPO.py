import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.maxlikelihood_abstraction_parallel import MaxLikelihoodAbstraction
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_deterministic.updater import Updater
import DPO.helper as helper
from DPO.helper import Helper
import os
import csv

problem = 'mass'
SINK = False
INIT_DETERMINISTIC_PARAM = np.array([-0.3, -0.3])
A = np.array([[1., 1.], [0., 1.]])
B = np.array([[0.], [1.]])
Q = np.diag([1., 0.])
R = 0.1 * np.eye(1)
GAMMA = 0.95
ENV_NOISE = 0.1 * np.eye(INIT_DETERMINISTIC_PARAM.size)

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
    for s in det_samples:
        # prev_action is the action that would be prescribed by the (not yet updated) deterministic policy.
        prev_action = deterministic_action(param, s[0])
        mcrst_provv = helper.get_mcrst(s[0], INTERVALS, SINK)
        mcrst = helper.get_index_from_mcrst(mcrst_provv, INTERVALS)
        if abs_opt_policy[mcrst] is not None:
            if prev_action in abs_opt_policy[mcrst]:  # prev_action is optimal.
                fictitious_samples.append([s[0], prev_action])
            else:  # we select the closest action to prev_action among the optimal ones.
                index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                fictitious_samples.append([s[0], abs_opt_policy[mcrst][index]])
    return fictitious_samples


def main(seed, args):
    alpha = 0.025 if args['alpha'] is None else args['alpha']
    lam = 0.0001 if args['lambda'] is None else args['lambda']
    N_MCRST_DYN = np.array([9, 9]) if args['mcrst'] is None else np.array([args['mcrst'], args['mcrst']])
    N_ITERATION = 120 if args['niter'] is None else args['niter']
    N_EPISODES = 500 if args['batch'] is None else args['batch']
    N_STEPS = 20 if args['nsteps'] is None else args['nsteps']
    file = False if args['file'] is None else args['file']

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

    if file:
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
    abstraction = MaxLikelihoodAbstraction(GAMMA, SINK, INTERVALS, lip_a_tf * STOCH_L_MULTIPLIER, env.Q, env.R,
                                           MAX_ACTION_VAL)
    abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)
    det_upd = Updater(help.getSeed())

    for i in range(0, N_ITERATION):

        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
        flat_samples = helper.flat_listoflists(determin_samples)

        # some operations are required in order to adapt the representation of \delta-MDP for the max-likelihood code.
        abstraction.divide_samples(flat_samples, problem, help.getSeed())
        abstraction.to_old_representation()
        abstraction.compute_abstract_tf(MIN_SPACE_VAL, MAX_SPACE_VAL, MAX_ACTION_VAL, ENV_NOISE)
        abstraction.to_new_representation()
        abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container(), reset=False)

        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, flat_samples, det_param, INTERVALS)
        det_param = det_upd.gradient_update(det_param, fictitious_samples, alpha, lam)
        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        # show the results of the iteration.
        print("Seed {} - Iteration N.{}".format(seed, i))
        print("Policy parameters: {}".format(det_param))
        print("Estimated performance measure: {}\n".format(estj))

        if file:
            file_writer.writerow([det_param[0][0], det_param[0][1], estj])

    if file:
        data_file.close()
