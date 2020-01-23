import gym, safety_gym
import potion.envs
import torch
from DPO.safetygym.REINFORCE.reinforce import reinforce
from DPO.safetygym.DPO.base_env import create_env
from DPO.safetygym.REINFORCE.cont_policies import ShallowGaussianPolicy
from potion.meta.steppers import ConstantStepper, RMSprop
from potion.common.logger import Logger
import numpy as np
import safety_gym.random_agent as rdm_ag


def feature_function(s):
    mask = np.array([True, True, False, True, True, False, False, True, True, True, False, True, True, False])
    # s = s.numpy()
    if len(s.shape) == 1:
        res = s[mask]

    else:
        res1 = []
        for t in s:
            res2 = []
            for u in t:
                res2.append(u[mask])
            res2 = torch.stack(res2, 0)
            res1.append(res2)
        res = torch.stack(res1, 0)

    return res


def main(env=None, seed=None):

    if env is None:
        env = create_env()

    gamma = 1  # TODO verify it!

    state_dim = 9  # dimensionality of the state space
    action_dim = 2  # dimensionality of the action space
    print(state_dim, action_dim)
    horizon = 500  # maximum length of a trajectory  TODO verify it!

    policy = ShallowGaussianPolicy(state_dim, #input size
                                   action_dim, #output size
                                   mu_init=torch.zeros(state_dim * action_dim), #initial mean parameters
                                   feature_fun=feature_function,
                                   logstd_init=-2.,
                                   learn_std=False
                              )

    # stepper = ConstantStepper(0.01)
    stepper = RMSprop()

    batchsize = 100
    log_dir = '../../logs'
    log_name = 'REINFORCE'
    logger = Logger(directory=log_dir, name=log_name)

    if seed is None:
        seed = 42

    env.seed(seed)

    # Reset the policy (in case is run multiple times)
    policy.set_from_flat(torch.zeros(state_dim * action_dim))

    reinforce(env = env,
              policy = policy,
              horizon = horizon,
              stepper = stepper,
              batchsize = batchsize,
              disc = gamma,
              iterations = 2001,
              seed = 42,
              logger = logger,
              save_params = 5, #Policy parameters will be saved on disk each 5 iterations
              shallow = True, #Use optimized code for shallow policies
              estimator = 'gpomdp', #Use the G(PO)MDP refined estimator
              baseline = 'peters' #Use Peter's variance-minimizing baseline
             )

    print(policy.get_flat())


if __name__ == '__main__':
    main()
