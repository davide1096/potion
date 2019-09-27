import gym
import potion.envs
from lqg1Dv2.abstraction import Abstraction
from lqg1Dv2.update import Updater
from lqg1Dv2.dynprog_updater import Updater
import lqg1Dv2.abstraction as abstr

INIT_DETERMINISTIC_PARAM = -1.
GAMMA = 0.9
LR_DET_POLICY = 0.1
N_ITERATIONS = 200

N_EPISODES = 200
N_STEPS = 20
N_EPISODES_ABSTRACT = 200
N_STEPS_ABSTRACT = 20

# INTERVALS = [[-2, -0.4], [-0.4, 0.1], [-0.1, 0], [0, 0.1], [0.1, 0.4], [0.4, 2]]

INTERVALS = [[-2, -1.6], [-1.6, -1.2], [-1.2, -1], [-1, -0.8], [-0.8, -0.6], [-0.6, -0.5], [-0.5, -0.4], [-0.4, -0.3],
             [-0.3, -0.2], [-0.2, -0.1], [-0.1, 0.], [0., 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5],
             [0.5, 0.6], [0.6, 0.8], [0.8, 1], [1, 1.2], [1.2, 1.6], [1.6, 2]]

env = gym.make('LQG1D-v0')
det_param = INIT_DETERMINISTIC_PARAM
abstraction = Abstraction(N_EPISODES_ABSTRACT, N_STEPS_ABSTRACT, INTERVALS)
# updater = Updater(len(INTERVALS), GAMMA, N_STEPS_ABSTRACT)
dp_updater = Updater(len(INTERVALS), GAMMA)


def deterministic_action(det_par, state):
    return det_par * state


def sampling_from_det_pol(envir, n_episodes, n_steps, det_par):
    samples_list = []
    for j in range(0, n_episodes):
        envir.reset()
        for k in range(0, n_steps):
            state = envir.get_state()
            action = deterministic_action(det_par, state)
            new_state, r, _, _ = env.step(action)
            samples_list.append([state[0], action[0], r, new_state[0]])
    return samples_list


def sampling_abstract_optimal_pol(abs_opt, mcrst):
    # todo
    return abs_opt[mcrst][0]


for i in range(0, N_ITERATIONS):
    deterministic_samples = sampling_from_det_pol(env, N_EPISODES, N_STEPS, det_param)
    abstraction.divide_samples(deterministic_samples)
    # abstraction.init_policy()
    # abstraction.compute_abstract_policy()
    # abstract_samples = abstraction.abstract_sampling()
    # update the abstract policy
    # abstract_policy = abstraction.get_abstract_policy()
    # updated_abstract_policy = updater.policy_gradient_update(abstract_samples, abstract_policy)
    abstract_optimal_policy = dp_updater.solve_mdp(abstraction.get_container())
    # abstraction.set_abstract_policy(updated_abstract_policy)
    # now the deterministic policy needs to be updated with the knowledge of the updated abstract policy
    # fictitious_samples = [[s[0], abstraction.draw_action_weighted_policy(abstr.get_mcrst(s[0], INTERVALS))]
    #                      for s in deterministic_samples]
    # update the det_param by minimizing the MSE loss function
    fictitious_samples = [[s[0], sampling_abstract_optimal_pol(abstract_optimal_policy,
                           abstr.get_mcrst(s[0], INTERVALS))] for s in deterministic_samples]
    for s in fictitious_samples:
        det_param -= LR_DET_POLICY * (det_param * s[0] - s[1]) * s[0]
    print("Updated deterministic policy parameter: {}\n".format(det_param))
