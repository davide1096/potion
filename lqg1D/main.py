import gym
import potion.envs
from lqg1D.policy import DeterministicPolicy as dp
from lqg1D.lqgspo import LqgSpo
import lqg1D.estimator as est
from lqg1D.abstract_mdp import AbstractMdp as AbsMdp
import lqg1D.abstract_mdp as abs_mdp

INIT_DETERMINISTIC_PARAM = -0.2
N_SAMPLES = 200
N_STEPS = 20
N_SAMPLES_ABSTRACT = 200
N_STEPS_ABSTRACT = 20


env = gym.make('LQG1D-v0')
# abstract_mdp contains the abstract policy, the abstract transition function and the abstract reward function.
abstract_fun = LqgSpo(env)
# det_pol is the deterministic policy from which we sample and the one that needs to be updated
det_pol = dp(INIT_DETERMINISTIC_PARAM)

# getting the samples according to the deterministic policy
samples = est.sampling_from_det_pol(env, N_SAMPLES, N_STEPS, det_pol)
# translating the samples with regard to macrostates
mcrst_samples = abstract_fun.from_states_to_macrostates(samples)

# update and visualize parameters for the abstract policies
abstract_fun.update_abs_policy(mcrst_samples)
abstract_fun.show_abs_policy_params()

# update and visualize parameters for the abstract transition functions
abstract_fun.update_abs_tf(mcrst_samples)
abstract_fun.show_abs_tf_params()

# getting the samples according to the abstract (stochastic) policy
abstract_mdp = AbsMdp(abstract_fun, -env.max_action, env.max_action)
abs_samples = abstract_mdp.sampling(N_SAMPLES_ABSTRACT, N_STEPS_ABSTRACT)

print("Initial states: ", abs_mdp.count_states([s[0] for s in abs_samples]))
print("Final states:   ", abs_mdp.count_states([s[3] for s in abs_samples]))