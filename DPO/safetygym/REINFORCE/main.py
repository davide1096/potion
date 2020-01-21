import gym, safety_gym
import potion.envs
import torch
from DPO.safetygym.REINFORCE.reinforce import reinforce
from DPO.safetygym.REINFORCE.cont_policies import ShallowGaussianPolicy
from potion.meta.steppers import ConstantStepper
from potion.common.logger import Logger
import numpy as np
import safety_gym.random_agent as rdm_ag


def main(env=None):

    if env is None:
        env = gym.make('Safexp-PointGoal0-v0')

    gamma = 1  # TODO verify it!

    state_dim = sum(env.observation_space.shape)  # dimensionality of the state space
    action_dim = sum(env.action_space.shape)  # dimensionality of the action space
    print(state_dim, action_dim)
    horizon = 500  # maximum length of a trajectory  TODO verify it!

    policy = ShallowGaussianPolicy(state_dim, #input size
                                   action_dim, #output size
                                   mu_init=torch.zeros(state_dim * action_dim), #initial mean parameters
                                   logstd_init=0.,
                                   learn_std=False
                              )

    state = torch.ones(state_dim)
    print(policy.act(state))
    print(policy.get_flat())

    stepper = ConstantStepper(0.01)

    batchsize = 50
    log_dir = '../../logs'
    log_name = 'REINFORCE'
    logger = Logger(directory=log_dir, name=log_name)

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
