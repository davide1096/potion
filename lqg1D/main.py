import gym
import potion.envs
from lqg1D.policy import DeterministicPolicy as dp
from lqg1D.lqgspo import LqgSpo
import lqg1D.estimator as est

INIT_DETERMINISTIC_PARAM = -0.2
N_SAMPLES = 200
N_STEPS = 20


env = gym.make('LQG1D-v0')
# abstract_mdp contains the abstract policy, the abstract transition function and the abstract reward function.
abstract_mdp = LqgSpo(env)
# det_pol is the deterministic policy from which we sample and the one that needs to be updated
det_pol = dp(INIT_DETERMINISTIC_PARAM)

samples = est.sampling_from_det_pol(env, N_SAMPLES, N_STEPS, det_pol)
abstract_mdp.update_abs_policy(samples)
abstract_mdp.show_abs_policy_params()