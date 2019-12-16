import gym
import numpy as np
from DPO.algorithm.abstraction.compute_atf.lipschitz_f_dads import LipschitzFdads
from DPO.algorithm.updater_abstract.updater import AbsUpdater
import DPO.algorithm.updater_deterministic.updater as det_upd
from DPO.visualizer.lqg1d_visualizer import Lqg1dVisualizer
import DPO.helper as helper

problem = 'cartpole1d'
SINK = True
INIT_DETERMINISTIC_PARAM = 30
GAMMA = 1
LIPSCHITZ_CONST_STATE = 0.5
LIPSCHITZ_CONST_ACTION = 0.5

N_ITERATION = 3000
N_EPISODES = 100
N_STEPS = 500

INTERVALS = [[-0.2093, -0.05], [-0.05, -0.04], [-0.04, -0.03], [-0.03, -0.02], [-0.02, -0.01], [-0.01, -0.005],
             [-0.005, 0.0], [0.0, 0.005], [0.005, 0.01], [0.01, 0.02], [0.02, 0.03], [0.03, 0.04], [0.04, 0.05],
             [0.05, 0.2093]]


# load and configure the environment.
env = gym.make('CartPole1d-v0')

# instantiate the components of the algorithm.
abstraction = LipschitzFdads(LIPSCHITZ_CONST_STATE, LIPSCHITZ_CONST_ACTION, GAMMA, SINK, INTERVALS)
abs_updater = AbsUpdater(GAMMA, SINK, INTERVALS)

visualizer = Lqg1dVisualizer("cartpole", "number of iterations", "parameter", " performance", "cartpole.jpg")
det_param = INIT_DETERMINISTIC_PARAM


def deterministic_action(det_par, state):
    return det_par * state


def sampling_from_det_pol(env, n_episodes, n_steps, det_par):
    samples_list = []
    for j in range(0, n_episodes):
        env.reset()
        k = 0
        single_sample = []
        done = False
        while k < n_steps and not done:
            state = env.state[2]
            action = [deterministic_action(det_par, state)]
            new_state, r, done, _ = env.step(action)
            single_sample.append([state.item(), action[0].item(), r, new_state.item()])
            k += 1

        # sink state. The action doesn't care, the important thing is taht it is different for every state.
        if k < n_steps:
            single_sample.append([new_state.item(), new_state.item(), 0, new_state.item()])

        samples_list.append(single_sample)
    return samples_list


# return for every state the action closest to the previous one among the actions in abs_opt_policy
def sampling_abstract_optimal_pol(abs_opt_policy, det_samples, param):

    fictitious_samples = []
    for sam in det_samples:
        single_sample = []
        for s in sam:

            # avoid to consider the sink state in the fictitious samples.
            if s[2] != 0:
                prev_action = deterministic_action(param, s[0])
                mcrst = helper.get_mcrst(s[0], INTERVALS, SINK)
                if prev_action in abs_opt_policy[mcrst]:
                    single_sample.append([s[0], prev_action])
                else:
                    index = np.argmin([abs(act - prev_action) for act in abs_opt_policy[mcrst]])
                    single_sample.append([s[0], abs_opt_policy[mcrst][index]])

        fictitious_samples.append(single_sample)
    return fictitious_samples


for i in range(0, N_ITERATION):
    deterministic_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
    abstraction.divide_samples(deterministic_samples, problem)
    abstraction.compute_abstract_tf()
    abstract_optimal_policy = abs_updater.solve_mdp(abstraction.get_container())

    fictitious_samples = sampling_abstract_optimal_pol(abstract_optimal_policy, deterministic_samples, det_param)
    det_param = det_upd.batch_gradient_update(det_param, fictitious_samples)
    # j = env.computeJ(det_param, ENV_NOISE, N_EPISODES)
    absj = helper.estimate_J_from_samples(deterministic_samples, GAMMA)

    print("Updated deterministic policy parameter: {}".format(det_param))
    print("Updated estimated performance measure: {}\n".format(absj))
    visualizer.show_values(det_param, absj, absj)

visualizer.save_image()

