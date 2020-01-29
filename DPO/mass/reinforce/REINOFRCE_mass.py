import gym
import torch
import potion.envs
from potion.actors.continuous_policies import ShallowGaussianPolicy
from DPO.mass.reinforce.reinforce_mass import reinforce
from potion.common.logger import Logger
from potion.meta.steppers import ConstantStepper


def main(seed=None, alpha=0.025, logsig=-3.):
    log_dir = '../logs'
    log_name = 'REINFORCE'
    logger = Logger(directory=log_dir, name=log_name)

    env = gym.make('mass-v0')

    state_dim = sum(env.observation_space.shape) #dimensionality of the state space
    action_dim = sum(env.action_space.shape) #dimensionality of the action space
    print(state_dim, action_dim)

    horizon = 20 #maximum length of a trajectory
    gamma = 0.95

    mu_init = torch.tensor([-0.3, -0.3])
    log_std_init = torch.tensor([logsig])
    learn_shallow_variance = False

    policy = ShallowGaussianPolicy(state_dim, #input size
                                   action_dim, #output size
                                   mu_init = mu_init, #initial mean parameters
                                   logstd_init = log_std_init, #log of standard deviation
                                   learn_std = learn_shallow_variance #We are NOT going to learn the variance parameter
                                  )

    print(policy.get_flat())

    stepper = ConstantStepper(alpha)

    batchsize = 500
    if seed is None:
        seed = 42

    env.seed(seed)

    if learn_shallow_variance:
        init_par = [log_std_init, mu_init]
        init_ten = torch.cat(init_par, 0)
    else:
        init_ten = mu_init

    # Reset the policy (in case is run multiple times)
    policy.set_from_flat(init_ten)

    stats, envj, estj = reinforce(alpha, logsig, env = env,
              policy = policy,
              horizon = horizon,
              stepper = stepper,
              batchsize = batchsize,
              disc = gamma,
              iterations = 300,
              seed = seed,
              logger = logger,
              save_params = 5, #Policy parameters will be saved on disk each 5 iterations
              shallow = True, #Use optimized code for shallow policies
              estimator = 'gpomdp', #Use the G(PO)MDP refined estimator
              baseline = 'peters' #Use Peter's variance-minimizing baseline
             )

    print(policy.get_flat())
    return stats, envj, estj
