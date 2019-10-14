import gym
import potion.envs
from lqg1Dv2.abstraction import Abstraction
from lqg1Dv2.dynprog_updater import Updater
import lqg1Dv2.abstraction as ab
import lqg1Dv2.visualization as vis
import random
import numpy as np

folder = "cartpole"
INIT_DETERMINISTIC_PARAM = -10
ENV_NOISE = 0
A = 1
B = 1
GAMMA = 0.9
LR_DET_POLICY = 0.1
N_ITERATION = 30
N_ITERATIONS_BATCH_GRAD = 100
BATCH_SIZE = 50

N_EPISODES = 100
N_STEPS = 50
N_EPISODES_ABSTRACT = 2000
N_STEPS_ABSTRACT = 20

INTERVALS = [[-0.21, -0.05], [-0.05, -0.04], [-0.04, -0.03], [-0.03, -0.02], [-0.02, -0.01], [-0.01, -0.005],
             [-0.005, 0.0], [0.0, 0.005], [0.005, 0.01], [0.01, 0.02], [0.02, 0.03], [0.03, 0.04], [0.04, 0.05],
             [0.05, 0.21]]


env = gym.make('ContCartPole-v0')
# print("Optimal value: ", env.computeOptimalK())
opt_par4visual = 0
det_param = INIT_DETERMINISTIC_PARAM
abstraction = Abstraction(N_EPISODES_ABSTRACT, N_STEPS_ABSTRACT, INTERVALS, A, B)
dp_updater = Updater(INTERVALS, GAMMA)
vis.initialization(A, B, opt_par4visual, ENV_NOISE, INIT_DETERMINISTIC_PARAM)


def deterministic_action(det_par, state):
    return det_par * state


def sampling_from_det_pol(env, n_episodes, n_steps, det_par):
    samples_list = []
    done = False
    for j in range(0, n_episodes):
        env.reset()
        k = 0
        while k < n_steps and not done:
            state = env.state[2]
            action = [deterministic_action(det_par, state)]
            new_state, r, done, _ = env.step(action)
            samples_list.append([state.item(), action[0].item(), r, new_state[2].item()])
            k += 1
    return samples_list


# return the action closest to the previous one among the optimal actions
def sampling_abstract_optimal_pol(abs_opt, st, param):
    prev_action = st * param
    mcrst = ab.get_mcrst(st, INTERVALS)
    if prev_action in abs_opt[mcrst]:
        return prev_action
    diff = min(abs(act - prev_action) for act in abs_opt[mcrst])
    return prev_action + diff if prev_action + diff in abs_opt[mcrst] else prev_action - diff


for i in range(0, N_ITERATION):
    deterministic_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
    abstraction.divide_samples(deterministic_samples)
    # to observe the min action sampled in each macrostate
    # print([min(c.keys()) for c in abstraction.get_container()])
    abstract_optimal_policy = dp_updater.solve_mdp(abstraction.get_container())
    # to observe the min action among the best actions in each macrostate
    # print([min(ab) for ab in abstract_optimal_policy])

    fictitious_samples = [[s[0], sampling_abstract_optimal_pol(abstract_optimal_policy,
                                                               s[0], det_param)] for s in deterministic_samples]
    for e in range(0, N_ITERATIONS_BATCH_GRAD):
        accumulator = 0
        for b in range(0, BATCH_SIZE):
            s = fictitious_samples[random.randint(0, len(fictitious_samples) - 1)]
            accumulator += (det_param * s[0] - s[1]) * s[0]
        det_param = det_param - LR_DET_POLICY * (accumulator / BATCH_SIZE)
    print("Updated deterministic policy parameter: {}\n".format(det_param))
    vis.show_new_value(det_param, opt_par4visual)
vis.save_img(ab.get_tf_known(), A, B, ENV_NOISE, folder)

