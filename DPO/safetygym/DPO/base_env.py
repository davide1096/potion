from safety_gym.envs.engine import Engine
import numpy as np


def create_env(seed=42):
    config_dict = {# 'observation_flatten': False,
                   'robot_base': 'xmls/point.xml',
                   'observe_goal_comp': True,
                   '_seed': seed
    }
    env = Engine(config=config_dict)
    return env


# env = create_env()
# reset_val = env.reset()
# print(reset_val)
# for _ in range(1000):
#     o, r, _, _ = env.step(env.action_space.sample())
#     # print(o['goal_lidar'])
#     print(o)
#     print("{}\n".format(r))


