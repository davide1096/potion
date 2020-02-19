from safety_gym.envs.engine import Engine


def create_env(seed=42):
    config_dict = {'robot_base': 'xmls/point.xml',
                   'observe_goal_comp': True,
                   '_seed': seed,
                   'num_steps': 2000
    }
    env = Engine(config=config_dict)
    return env

