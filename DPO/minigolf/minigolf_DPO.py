import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_abstract.bounded_mdp.IVI import IVI
import DPO.helper as helper
from DPO.helper import Helper
from DPO.minigolf.RBFNet import RBFNet
import csv
import os


# task parameters.
problem = 'minigolf'
SINK = True
ENV_NOISE = 0
GAMMA = 0.99
MIN_SPACE_VAL = [0]
MAX_SPACE_VAL = [20]

# radial bases configuration.
CENTERS = [4, 8, 12, 16]
STD_DEV = 4
INIT_W = [1, 1, 1, 1]


def deterministic_action(state, rbf):
    if np.all(state[0] < 0):  # there is no meaning in prescribing an action for a state < 0.
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
            single_sample.append([state[0], action[0], r, new_state[0]])
            k += 1
        samples_list.append(single_sample)
    return samples_list


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, rbf, INTERVALS):
    fictitious_samples = []
    for s in det_samples:
        if s[0] > 0:  # exclude s<0 from fictitious samples. They are only collected to detect the end of an episode.
            # prev_action is the action that would be prescribed by the (not yet updated) deterministic policy.
            prev_action = deterministic_action(np.reshape(s[0], (1, 1)), rbf)
            prev_action = prev_action[0]
            mcrst = helper.get_mcrst(s[0], INTERVALS, SINK)
            index_mcrst = helper.get_index_from_mcrst(mcrst, INTERVALS)
            if index_mcrst in abs_opt_policy.keys():
                if helper.array_in(prev_action, abs_opt_policy[index_mcrst]):  # prev_action is optimal.
                    fictitious_samples.append([s[0], prev_action])
                else:  # we select the closest action to prev_action among the optimal ones.
                    index = np.argmin([helper.arr_distance(act, prev_action) for act in abs_opt_policy[index_mcrst]])
                    fictitious_samples.append([s[0], abs_opt_policy[index_mcrst][index]])
    return fictitious_samples


def main(seed, args):
    alpha = 0.001 if args['alpha'] is None else args['alpha']
    lam = 0.0005 if args['lambda'] is None else args['lambda']
    N_MCRST_DYN =[12] if args['mcrst'] is None else [args['mcrst']]
    N_ITERATION = 700 if args['niter'] is None else args['niter']
    N_EPISODES = 500 if args['batch'] is None else args['batch']
    N_STEPS = 20 if args['nsteps'] is None else args['nsteps']
    LDELTAS = 0 if args['Lds'] is None else args['Lds']
    file = False if args['file'] is None else args['file']

    help = Helper(seed)
    env = gym.make('ComplexMiniGolf-v0')  # load and configure the environment.
    env.sigma_noise = ENV_NOISE
    env.gamma = GAMMA
    env.seed(help.getSeed())
    total_failures = 0
    rbf = RBFNet(CENTERS, STD_DEV, INIT_W, help.getSeed(), alpha, lam)

    if file:
        filename = "../csv/minigolf/DPO/ALPHA={}/LAM={}/data{}.csv".format(alpha, lam, help.getSeed())
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data_file = open(filename, mode='w')
        file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['w1', 'w2', 'w3', 'w4', 'tot_failures', 'estj'])

    INTERVALS = helper.get_constant_intervals(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN)
    print("Seed: {} - Alpha: {}, Lambda: {}".format(seed, alpha, lam))
    print("INTERVALS: {}\n{}\n".format(N_MCRST_DYN, INTERVALS))
    INTERVALS[0] = [[-4, 0]] + INTERVALS[0]  # add the first macrostate representing the goal.

    for i in range(N_ITERATION):

        determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, rbf)
        flat_samples = helper.flat_listoflists(determin_samples)

        if i == 0:
            # instantiate the components of the algorithm.
            abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS)
            abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS, -100) if LDELTAS == 0 else \
                IVI(GAMMA, SINK, True, INTERVALS, -100)

        # build the \gamma-MDP.
        abstraction.divide_samples(flat_samples, problem, help.getSeed())
        abstraction.compute_abstract_tf(LDELTAS)
        # compute the value iteration.
        if LDELTAS == 0:  # standard VI if LDELTAS=0, otherwise we need the bounded-MDP.
            abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container(), reset=False)
        else:
            # some operations are required in order to adapt the representation of \delta-MDP for the bounded-MDP code.
            abstraction.to_old_representation()
            abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())
            abstraction.to_new_representation(change_tf=False)
            abs_opt_pol_dict = {}
            for j, aop in enumerate(abs_opt_pol):
                abs_opt_pol_dict[j] = aop
            abs_opt_pol = abs_opt_pol_dict

        # project back the policy.
        fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, flat_samples, rbf, INTERVALS)
        rbf.fit(fictitious_samples)
        estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

        # show the results of the iteration.
        print("Seed {} - Iteration N.{}".format(seed, i))
        print("RBF weights: {}".format(rbf.w))
        print("Estimated performance measure: {}".format(estj))
        hundred, failing_states = helper.minigolf_reward_counter(flat_samples)
        print("Failing states: {}".format(failing_states))
        total_failures += hundred
        print("Cumulative fails: {}\n".format(total_failures))

        w = rbf.w
        if file:
            file_writer.writerow([w[0], w[1], w[2], w[3], total_failures, estj])

    if file:
        data_file.close()

