import gym
import potion.envs
from lqg1Dv2.abstraction import Abstraction
from lqg1Dv2.dynprog_updater import Updater
import lqg1Dv2.abstraction as ab
import lqg1Dv2.visualization as vis
from lqg1Dv2.abstract_tf import AbstractTF
import random
import numpy as np

folder = "lqg1d"
INIT_DETERMINISTIC_PARAM = -0.1
ENV_NOISE = 0
A = 1
B = 1
GAMMA = 0.9
LR_DET_POLICY = 0.1
N_ITERATION = 30
N_ITERATIONS_BATCH_GRAD = 200
BATCH_SIZE = 50

N_EPISODES = 2000
N_STEPS = 20

N_EPISODES_ABSTRACT = 2000
N_STEPS_ABSTRACT = 20

# INTERVALS = [[-2, -1], [-1, -0.5], [-0.5, 0], [0, 0.5], [0.5, 1], [1, 2]]

# INTERVALS = [[-2, -0.4], [-0.4, -0.1], [-0.1, 0], [0, 0.1], [0.1, 0.4], [0.4, 2]]

INTERVALS = [[-2, -1.6], [-1.6, -1.2], [-1.2, -1], [-1, -0.8], [-0.8, -0.6], [-0.6, -0.5], [-0.5, -0.4], [-0.4, -0.3],
             [-0.3, -0.2], [-0.2, -0.1], [-0.1, 0.], [0., 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5],
             [0.5, 0.6], [0.6, 0.8], [0.8, 1], [1, 1.2], [1.2, 1.6], [1.6, 2]]

# INTERVALS = [[-2, -1.95], [-1.95, -1.9], [-1.9, -1.85], [-1.85, -1.8], [-1.8, -1.75], [-1.75, -1.7], [-1.7, -1.65],
#              [-1.65, -1.6], [-1.6, -1.55], [-1.55, -1.5], [-1.5, -1.45], [-1.45, -1.4], [-1.4, -1.35], [-1.35, -1.3],
#              [-1.3, -1.25], [-1.25, -1.2], [-1.2, -1.15], [-1.15, -1.1], [-1.1, -1.05], [-1.05, -1.0], [-1.0, -0.95],
#              [-0.95, -0.9], [-0.9, -0.85], [-0.85, -0.8], [-0.8, -0.75], [-0.75, -0.7], [-0.7, -0.65], [-0.65, -0.6],
#              [-0.6, -0.55], [-0.55, -0.5], [-0.5, -0.45], [-0.45, -0.4], [-0.4, -0.35], [-0.35, -0.3], [-0.3, -0.25],
#              [-0.25, -0.2], [-0.2, -0.15], [-0.15, -0.1], [-0.1, -0.05], [-0.05, 0.0], [0.0, 0.05], [0.05, 0.1],
#              [0.1, 0.15], [0.15, 0.2], [0.2, 0.25], [0.25, 0.3], [0.3, 0.35], [0.35, 0.4], [0.4, 0.45], [0.45, 0.5],
#              [0.5, 0.55], [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 0.75], [0.75, 0.8], [0.8, 0.85], [0.85, 0.9],
#              [0.9, 0.95], [0.95, 1.0], [1.0, 1.05], [1.05, 1.1], [1.1, 1.15], [1.15, 1.2], [1.2, 1.25], [1.25, 1.3],
#              [1.3, 1.35], [1.35, 1.4], [1.4, 1.45], [1.45, 1.5], [1.5, 1.55], [1.55, 1.6], [1.6, 1.65], [1.65, 1.7],
#              [1.7, 1.75], [1.75, 1.8], [1.8, 1.85], [1.85, 1.9], [1.9, 1.95], [1.95, 2.0]]


env = gym.make('LQG1D-v0')
env.sigma_noise = ENV_NOISE
env.A = np.array([A]).reshape((1, 1))
env.B = np.array([B]).reshape((1, 1))
print("Optimal value: ", env.computeOptimalK())
opt_par4visual = round(env.computeOptimalK()[0][0], 3)
det_param = INIT_DETERMINISTIC_PARAM
abstraction = Abstraction(N_EPISODES_ABSTRACT, N_STEPS_ABSTRACT, INTERVALS, A, B)
dp_updater = Updater(INTERVALS, GAMMA)
vis.initialization(A, B, opt_par4visual, ENV_NOISE, INIT_DETERMINISTIC_PARAM)


def deterministic_action(det_par, state):
    return det_par * state


def sampling_from_det_pol(env, n_episodes, n_steps, det_par):
    samples_list = []
    for j in range(0, n_episodes):
        env.reset()
        for k in range(0, n_steps):
            state = env.get_state()
            action = deterministic_action(det_par, state)
            new_state, r, _, _ = env.step(action)
            samples_list.append([state[0], action[0], r, new_state[0]])
    return samples_list


# return the action closest to the previous one among the optimal actions
def sampling_abstract_optimal_pol(abs_opt, st, param):
    prev_action = st * param
    mcrst = ab.get_mcrst(st, INTERVALS)
    if prev_action in abs_opt[mcrst]:
        return prev_action
    diff = min(abs(act - prev_action) for act in abs_opt[mcrst])
    return prev_action + diff if prev_action + diff in abs_opt[mcrst] else prev_action - diff


def compute_performance(det_samples, det_param):
    return -0.5 * (1 + det_param * det_param) * sum([s[0] * s[0] for s in det_samples]) / len(det_samples)


for i in range(0, N_ITERATION):
    deterministic_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
    abstraction.divide_samples(deterministic_samples)
    # n_actions = abstraction.count_actions()
    # abstract_tf_solver = AbstractTF(abstraction.get_container(), 7.5, INTERVALS)
    # abstract_tf, id_actions = abstract_tf_solver.get_abstract_tf()
    # abstraction.set_abstract_tf(abstract_tf, id_actions)
    # to observe the min action sampled in each macrostate
    # print([min(c.keys()) for c in abstraction.get_container() if len(list(c.keys())) > 0])
    abstract_optimal_policy = dp_updater.solve_mdp(abstraction.get_container())
    # to observe the min action among the best actions in each macrostate
    # print([min(ab) for ab in abstract_optimal_policy if ab is not None])

    fictitious_samples = [[s[0], sampling_abstract_optimal_pol(abstract_optimal_policy,
                                                               s[0], det_param)] for s in deterministic_samples]
    for e in range(0, N_ITERATIONS_BATCH_GRAD):
        accumulator = 0
        for b in range(0, BATCH_SIZE):
            s = fictitious_samples[random.randint(0, len(fictitious_samples) - 1)]
            accumulator += (det_param * s[0] - s[1]) * s[0]
        det_param = det_param - LR_DET_POLICY * (accumulator / BATCH_SIZE)
    j = compute_performance(deterministic_samples, det_param)
    print("Updated deterministic policy parameter: {}\n".format(det_param))
    print("Updated performance measure: {}\n".format(j))
    vis.show_new_value(det_param, opt_par4visual, j)
vis.save_img(ab.get_tf_known(), A, B, ENV_NOISE, folder)

