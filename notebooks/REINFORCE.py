import gym
import torch
import potion.envs
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.algorithms.reinforce import reinforce
from potion.common.logger import Logger
from potion.meta.steppers import ConstantStepper

log_dir = '../logs'
log_name = 'REINFORCE'
logger = Logger(directory=log_dir, name = log_name)

env = gym.make('ContCartPole-v0')

state_dim = sum(env.observation_space.shape) #dimensionality of the state space
action_dim = sum(env.action_space.shape) #dimensionality of the action space
print(state_dim, action_dim)

horizon = 500 #maximum length of a trajectory
gamma = 1.

policy = ShallowGaussianPolicy(state_dim, #input size
                               action_dim, #output size
                               mu_init = torch.zeros(4), #initial mean parameters
                               logstd_init = 0., #log of standard deviation
                               learn_std = False #We are NOT going to learn the variance parameter
                              )

# state = torch.ones(4)
# print(policy.act(state))

print(policy.get_flat())

stepper = ConstantStepper(0.05)

batchsize = 100
seed = 42

env.seed(seed)

policy.set_from_flat(torch.zeros(4)) #Reset the policy (in case is run multiple times)

reinforce(env = env,
          policy = policy,
          horizon = horizon,
          stepper = stepper,
          batchsize = batchsize,
          disc = gamma,
          iterations = 75,
          seed = 42,
          logger = logger,
          save_params = 5, #Policy parameters will be saved on disk each 5 iterations
          shallow = True, #Use optimized code for shallow policies
          estimator = 'gpomdp', #Use the G(PO)MDP refined estimator
          baseline = 'peters' #Use Peter's variance-minimizing baseline
         )

print(policy.get_flat())