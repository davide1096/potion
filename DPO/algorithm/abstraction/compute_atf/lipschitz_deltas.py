from DPO.algorithm.abstraction.lipschitz_abstraction import LipschitzAbstraction
import DPO.algorithm.abstraction.compute_atf.abstract_tf.sample_distribution as sample_dist
import DPO.algorithm.abstraction.compute_atf.abstract_tf.bounded_atf as bounded_atf
import DPO.helper as helper
import numpy as np
import logging
# import DPO.visualizer.bounds_visualizer as bvis
import gym
import potion.envs


class LipschitzDeltaS(LipschitzAbstraction):

    def __init__(self, gamma, sink, intervals=None, ls=None, la=None, Q=None, R=None, maxa_env=1):
        super().__init__(gamma, sink, intervals, Q, R, maxa_env)
        if ls is not None:
            self.LIPSCHITZ_CONST_S = ls
        if la is not None:
            self.LIPSCHITZ_CONST_A = la

    # ds0 is True when the hypothesis of deltaS = 0 is valid.
    # It means that taking the same action in different states will produce the same delta s (deltas = s' - s).
    def calculate_single_atf(self, mcrst, act, ds0, mins_env, maxs_env, maxa_env, std=0):

        # Used to check the correct new state-estimate for unvisited (s, a) pairs.
        # ENV_NOISE = 0 * np.eye(2)
        # A = np.array([[1., 1.], [0., 1.]])
        # B = np.array([[0.], [1.]])
        # GAMMA = 0.95
        # env = gym.make('mass-v0')
        # env.sigma_noise = ENV_NOISE
        # env.A = A
        # env.B = B
        # env.gamma = GAMMA
        # env.seed(0)

        cont = self.container[mcrst]
        new_state_bounds = []
        delta_s = cont[act]['new_state'] - cont[act]['state']
        act_clip = np.clip(act, -maxa_env, maxa_env)

        for action in cont.keys():

            action_clip = np.clip(action, -maxa_env, maxa_env)
            dist_s_shat = abs(cont[act]['state'] - cont[action]['state'])
            # I calculate the difference I can have taking act in a diff state according to the Lipschitz hyp on deltas.
            bound1 = abs(np.dot(self.LIPSCHITZ_CONST_S, dist_s_shat))
            min_val1 = cont[action]['state'] + delta_s - bound1
            max_val1 = cont[action]['state'] + delta_s + bound1
            min_val1 = np.clip(min_val1, mins_env, maxs_env)
            max_val1 = np.clip(max_val1, mins_env, maxs_env)

            bounds = [[np.round(min_val1, 3), np.round(max_val1, 3)]]

            # if not ds0 the bound computed above is not a single point.
            if not ds0:
                # I compute a bound according to the distance between two diff actions.
                dist_a_ahat = np.array([abs(action_clip - act_clip)])
                delta_s2 = cont[action]['new_state'] - cont[action]['state']
                # the bound is the difference I can have when I take act instead of action
                # according to the Lipschitz hypothesis on delta s.
                bound2 = abs(np.dot(self.LIPSCHITZ_CONST_A, dist_a_ahat))
                min_val2 = cont[action]['state'] + delta_s2 - bound2
                max_val2 = cont[action]['state'] + delta_s2 + bound2
                min_val2 = np.clip(min_val2, mins_env, maxs_env)
                max_val2 = np.clip(max_val2, mins_env, maxs_env)

                bounds.append([np.round(min_val2, 3), np.round(max_val2, 3)])

            min_val, max_val = helper.interval_intersection(bounds)
            # in case of void intersections, None values are returned.
            if min_val is not None and max_val is not None:
                new_state_bounds.append([min_val, max_val])

            # Used to check the correct new state-estimate for unvisited (s, a) pairs.
            # env.reset(cont[action]['state'])
            # ns, _, _, _ = env.step(act)
            # if np.any(np.round(min_val, 2) - np.round(ns, 2) > 0.02) or np.any(np.round(ns, 2) - np.round(max_val, 2) > 0.02):
            #     print("here")

            # use it only when you need to plot bounds
            # else:
            #     new_state_bounds.append([0, 0])

        # --- matplot ---
        # if mcrst == 0 and act == min(list(self.container[0].keys())):
        #     true_value = []
        #     for action in cont.keys():
        #         true_value.append(self.a * cont[action]['state'] + self.b * act)
        #     bvis.plot_bounds(new_state_bounds, "min action", true_value)
        # if mcrst == 0 and act == max(list(self.container[0].keys())):
        #     true_value = []
        #     for action in cont.keys():
        #         true_value.append(self.a * cont[action]['state'] + self.b * act)
        #     bvis.plot_bounds(new_state_bounds, "max action", true_value)
        # ---------------

        if ds0:
            return sample_dist.abstract_tf(self.intervals, new_state_bounds, self.sink)
        else:
            return bounded_atf.abstract_tf(self.intervals, new_state_bounds, self.sink)
