import gym
import potion.envs
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_deltas import LipschitzDeltaS
from DPO.algorithm.updater_abstract.updater import AbsUpdater
from DPO.algorithm.updater_abstract.bounded_mdp.IVI import IVI
import DPO.helper as helper
import logging
from DPO.minigolf.RBFNet import RBFNet

problem = 'lqg1d'
SINK = False
ENV_NOISE = 0
A = 1
B = 1
GAMMA = 0.9
# optA = when we consider the problem lipschitz 0 wrt deltas hypothesis (bounded by a distance among states).
# Set optA = 0 to use the standard algorithm.
optA = 0
LIPSCHITZ_CONST_STATE = A
LIPSCHITZ_CONST_ACTION = B
LIPSCHITZ_STOCH_ATF = B

N_ITERATION = 5000
N_EPISODES = 2000
N_STEPS = 20

INTERVALS = [[-2, -1.6], [-1.6, -1.2], [-1.2, -0.8], [-0.8, -0.5], [-0.5, -0.3], [-0.3, -0.1], [-0.1, 0.1],
             [0.1, 0.3], [0.3, 0.5], [0.5, 0.8], [0.8, 1.2], [1.2, 1.6], [1.6, 2]]


# load and configure the environment.
env = gym.make('LQG1D-v0')
env.sigma_noise = ENV_NOISE
env.A = np.array([A]).reshape((1, 1))
env.B = np.array([B]).reshape((1, 1))
env.gamma = GAMMA
env.seed(helper.SEED)

# calculate the optimal values of the problem.
opt_par4vis = round(env.computeOptimalK()[0][0], 3)
optJ4vis = round(env.computeJ(env.computeOptimalK(), 0, N_EPISODES), 3)
logging.basicConfig(level=logging.DEBUG, filename='../test.log', filemode='w', format='%(message)s')

# instantiate the components of the algorithm.
abstraction = LipschitzDeltaS(GAMMA, SINK, INTERVALS, A, B)
# abstraction = MaxLikelihoodAbstraction(GAMMA, SINK, INTERVALS, B)

abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS) if optA else IVI(GAMMA, SINK, True, INTERVALS)
# abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)
rbf = RBFNet([-1.32, -0.66, 0, 0.66, 1.32], [0.5, 0.2, 0.05, -0.2, -0.5], lr=0.1, epochs=200)


def deterministic_action(state):
    return rbf.predict(state)[0]


def sampling_from_det_pol(env, n_episodes, n_steps):
    samples_list = []
    for i in range(0, n_episodes):
        env.reset()
        single_sample = []
        for j in range(0, n_steps):
            state = env.get_state()
            action = deterministic_action(state)
            new_state, r, _, _ = env.step(action)
            single_sample.append([state[0], action[0], r, new_state[0]])
        samples_list.append(single_sample)
    return samples_list


def sampling_abstract_optimal_pol(abs_opt_policy, det_samples):
    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:
            prev_action = deterministic_action(np.reshape(s[0], (1, 1)))
            prev_action = prev_action[0]
            mcrst = helper.get_mcrst(s[0], INTERVALS, SINK)
            if prev_action in abs_opt_policy[mcrst]:
                single_sample.append([s[0], prev_action])
            else:
                index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                single_sample.append([s[0], abs_opt_policy[mcrst][index]])
        fictitious_samples.append(single_sample)
    return fictitious_samples


for i in range(0, N_ITERATION):
    determin_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS)
    abstraction.divide_samples(determin_samples, problem)
    abstraction.compute_abstract_tf(optA, ENV_NOISE)

    abs_opt_pol = abs_updater.solve_mdp(abstraction.get_container())
    # logging.debug([min(a) for a in abstract_optimal_policy])
    # logging.debug("\n")
    logging.debug("Optimal policy: {}".format(abs_opt_pol))

    fictitious_samples = sampling_abstract_optimal_pol(abs_opt_pol, determin_samples)
    fictitious_samples = helper.flat_listoflists(fictitious_samples)
    X = np.reshape([f[0] for f in fictitious_samples], (len(fictitious_samples),))
    y = np.reshape([f[1] for f in fictitious_samples], (len(fictitious_samples),))
    rbf.fit(X, y)
    estj = helper.estimate_J_from_samples(determin_samples, GAMMA)

    print("Iteration n.{}".format(i))
    print("W: {}".format(rbf.w))
    print("b: {}".format(rbf.b))
    print("Updated estimated performance measure: {}".format(estj))
