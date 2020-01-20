from safety_gym.envs.engine import Engine


def create_env(seed):
    config_dict = {# 'observation_flatten': False,
                   'robot_base': 'xmls/point.xml',
                   'sensors_obs': ['accelerometer', 'velocimeter', 'gyro'],
                   'observe_goal_lidar': True,
                   'lidar_num_bins': 4,
                   '_seed': seed
    }
    env = Engine(config=config_dict)
    return env

# print(env.reset())
# for _ in range(100):
#     o, _, _, _ = env.step(env.action_space.sample())
#     print(o)
