from safety_gym.envs.engine import Engine
import safety_gym.random_agent as rdm_ag
from DPO.safetygym.REINFORCE.main import main

config_dict = {# 'observation_flatten': False,
               'robot_base': 'xmls/point.xml',
               'sensors_obs': [],
               'observe_goal_comp': True}
env = Engine(config=config_dict)

# main(env)
print(env.reset())
# for i in range(10):
#     obs, reward, done, info = env.step(1)
#     print(obs)

# rdm_ag.run_random(env)


