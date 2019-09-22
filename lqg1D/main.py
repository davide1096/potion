import gym
import potion.envs
from lqg1D.policy import DeterministicPolicy as dp
from lqg1D.lqgspo import LqgSpo
import lqg1D.estimator as est
from lqg1D.abstract_mdp import AbstractMdp as AbsMdp
import lqg1D.abstract_mdp as abs_mdp

INIT_DETERMINISTIC_PARAM = -0.1
LR_DET_POLICY = 0.01
N_ITERATIONS = 2000

N_SAMPLES = 200
N_STEPS = 20
N_SAMPLES_ABSTRACT = 200
N_STEPS_ABSTRACT = 20


env = gym.make('LQG1D-v0')
deterministic_policy_par = INIT_DETERMINISTIC_PARAM
# abstract_fun contains the abstract policy, the abstract transition function and the abstract reward function.
abstract_fun = LqgSpo(env)
# det_pol is the deterministic policy from which we get samples and the one that needs to be updated
det_pol = dp(deterministic_policy_par)
abstract_mdp = AbsMdp(abstract_fun, -env.max_action, env.max_action, N_SAMPLES_ABSTRACT, N_STEPS_ABSTRACT)

for i in range(0, N_ITERATIONS):
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
    # show the transition probability between macrostates related to the mean action in every macrostate
    abstract_fun.show_tf_prob()

    # getting the samples according to the abstract (stochastic) policy
    abs_samples = abstract_mdp.sampling()

    # update the abstract policy with abs_samples
    abstract_mdp.policy_gradient_update(abs_samples)
    abstract_fun.show_abs_policy_params()
    abstract_mdp.show_critic_vparams()

    # now the goal is to project back the abstract policy (updated) in the original policy
    if not abstract_fun.abstract_policy_version:
        fictitious_samples = [[s[0], abstract_mdp.draw_action_gaussian_policy(abstract_fun.get_mcrst(s[0]))]
                              for s in samples]
    else:
        fictitious_samples = [[s[0], abstract_mdp.draw_action_weighted_policy(abstract_fun.get_mcrst(s[0]))]
                              for s in samples]

    # updating the deterministic_policy_par minimizing the MSE loss function
    for s in fictitious_samples:
        deterministic_policy_par -= LR_DET_POLICY * (deterministic_policy_par * s[0] - s[1]) * s[0]

    det_pol.update_param(deterministic_policy_par)
    print("Updated deterministic policy parameter: {}\n".format(deterministic_policy_par))
