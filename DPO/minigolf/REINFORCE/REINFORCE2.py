import gym
import potion.envs
import torch
from DPO.minigolf.REINFORCE.radial_basis_policy import RadialBasisPolicy
from DPO.minigolf.REINFORCE.reinforce2 import reinforce2
from potion.meta.steppers import ConstantStepper
from potion.common.logger import Logger
import numpy as np


def feature_function(s):
    sigma = 4
    centers = [4, 8, 12, 16]
    res = [np.exp(-1 / (2 * sigma ** 2) * (s - c) ** 2) for c in centers]
    cat_dim = len(s.shape)
    res = torch.cat(res, cat_dim - 1)
    return res


def main(seed=None):

    gamma = 0.99
    env = gym.make('MiniGolf-v0')
    env.sigma_noise = 0
    env.gamma = gamma

    state_dim = sum(env.observation_space.shape)  # dimensionality of the state space
    action_dim = sum(env.action_space.shape)  # dimensionality of the action space
    print(state_dim, action_dim)
    horizon = 20  # maximum length of a trajectory

    mu_init = torch.tensor([1., 1., 1., 1.])
    log_std_init = torch.tensor([-3.])

    policy = RadialBasisPolicy(state_dim, #input size
                                   action_dim, #output size
                                   mu_init=mu_init, #initial mean parameters
                                   feature_fun=feature_function,
                                   logstd_init=log_std_init,
                                   learn_std=True
                              )

    stepper = ConstantStepper(0.01)

    batchsize = 500
    log_dir = '../../../logs'
    log_name = 'REINFORCE'
    logger = Logger(directory=log_dir, name=log_name)

    if seed is None:
        seed = 42

    env.seed(seed)

    init_par = [log_std_init, mu_init]
    init_ten = torch.cat(init_par, 0)

    # Reset the policy (in case is run multiple times)
    policy.set_from_flat(init_ten)

    stats = reinforce2(env = env,
              policy = policy,
              horizon = horizon,
              stepper = stepper,
              batchsize = batchsize,
              disc = gamma,
              iterations = 501,
              seed = seed,
              logger = logger,
              save_params = 5, #Policy parameters will be saved on disk each 5 iterations
              shallow = True, #Use optimized code for shallow policies
              estimator = 'gpomdp', #Use the G(PO)MDP refined estimator
              baseline = 'peters' #Use Peter's variance-minimizing baseline
             )

    policy.get_flat()
    return stats
