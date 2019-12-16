import gym
import potion.envs
import torch
from lqg1Dscalable.mg_reinforce_rb.radial_basis_policy import RadialBasisPolicy
from lqg1Dscalable.mg_reinforce_rb.reinforce2 import reinforce2
from potion.meta.steppers import ConstantStepper
from potion.common.logger import Logger
import numpy as np

env = gym.make('MiniGolf-v0')
env.sigma_noise = 0

state_dim = sum(env.observation_space.shape) #dimensionality of the state space
action_dim = sum(env.action_space.shape) #dimensionality of the action space
print(state_dim, action_dim)
horizon = 500 #maximum length of a trajectory
gamma = 1.


def feature_function(s):
    sigma = 3
    centers = [3, 6, 10, 14, 17, 0]
    bias = [0, 0, 0, 0, 0, 1]
    res = [(1 - b) * np.exp(-1 / (2 * sigma ** 2) * (s - c) ** 2) + b for c, b in zip(centers, bias)]
    cat_dim = len(s.shape)
    res = torch.cat(res, cat_dim - 1)
    return res


mu_init = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 1])
log_std_init = torch.tensor([0.])


policy = RadialBasisPolicy(state_dim, #input size
                               action_dim, #output size
                               mu_init=mu_init, #initial mean parameters
                               feature_fun=feature_function,
                               logstd_init=log_std_init,
                               learn_std=True
                          )

state = torch.ones(1)
policy.act(state)
policy.get_flat()

stepper = ConstantStepper(0.005)

batchsize = 100
log_dir = '../../logs'
log_name = 'REINFORCE'
logger = Logger(directory=log_dir, name = log_name)

seed = 42
env.seed(seed)

init_par = [log_std_init, mu_init]
init_ten = torch.cat(init_par, 0)

policy.set_from_flat(init_ten) #Reset the policy (in case is run multiple times)

reinforce2(env = env,
          policy = policy,
          horizon = horizon,
          stepper = stepper,
          batchsize = batchsize,
          disc = gamma,
          iterations = 750,
          seed = 42,
          logger = logger,
          save_params = 5, #Policy parameters will be saved on disk each 5 iterations
          shallow = True, #Use optimized code for shallow policies
          estimator = 'gpomdp', #Use the G(PO)MDP refined estimator
          baseline = 'peters' #Use Peter's variance-minimizing baseline
         )

policy.get_flat()
