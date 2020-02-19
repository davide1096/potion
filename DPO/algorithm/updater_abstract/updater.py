# to avoid a slow computation.
MAX_ITERATIONS = 500
EPSILON = 0.0001


class AbsUpdater(object):

    def __init__(self, gamma, sink, intervals=None, sink_val=None):
        super().__init__()
        self.intervals = intervals
        self.gamma = gamma
        self.sink = sink
        self.v_function = {}
        self.best_policy = {}
        self.sink_val = sink_val

    def solve_mdp(self, container, reset=True):

        if len(self.v_function) == 0:
            self.v_function = {}
            self.best_policy = {}
            for k in container.keys():
                self.v_function[k] = 0
                self.best_policy[k] = []

        new_vf = self.single_step_update(container, reset)
        n_iterations = 0

        while not self.solved(new_vf) and n_iterations < MAX_ITERATIONS:
            self.v_function = new_vf
            new_vf = self.single_step_update(container, reset)
            n_iterations += 1

        self.v_function = new_vf
        return self.best_policy

    def solved(self, new):
        for k in new.keys():
            if abs(new[k] - self.v_function[k]) > EPSILON:
                return False
        return True

    def single_step_update(self, container, reset):

        new_v_function = {}
        for k in container.keys():
            new_v_function[k] = 0 if reset else self.v_function[k]

        # k1 --> mcrst index
        for k1, v1 in container.items():
            if k1 != 'sink':

                possible_actions = {}

                # k2 --> id of the single sample stored into the mcrst
                for k2, v2 in v1.items():
                    abs_reward = v2['abs_reward']

                    if 'abs_tf' in v2:
                        abs_tf = v2['abs_tf']
                        x = 0

                        # k3 --> mcrst index
                        for k3, v3 in abs_tf.items():
                            if k3 == 'sink':
                                x += v3 * self.sink_val
                            elif k3 in self.v_function.keys():
                                x += v3 * self.v_function[k3]
                            else:  # penalization for unexplored regions.
                                x += v3 * min(self.v_function.values())
                        possible_actions[k2] = abs_reward + self.gamma * x

                self.best_policy[k1], new_v_function[k1] = self.best_actions(possible_actions, k1, container)

        return new_v_function

    def best_actions(self, possibilities, i, container):

        if len(possibilities.items()) > 0:
            # target is the value of the v_function of the macrostate at this iteration.
            target = max(possibilities.values())
            keys = [k for k in possibilities.keys() if possibilities[k] == target]
            best_acts = [container[i][k]['action'] for k in keys]
            return best_acts, target

        else:
            return None, self.v_function[i]
