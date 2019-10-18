import lqg1Dscalable.helper as helper


class Abstraction(object):

    def __init__(self, intervals=None):
        super().__init__()
        self.intervals = intervals
        self.container = []

    def init_container(self):
        container = []
        for i in range(0, len(self.intervals)):
            container.append({})
        return container

    def get_container(self):
        return self.container

    def divide_samples(self, samples, problem, intervals=None):

        if intervals is not None:
            self.intervals = intervals

        # container is an array of dictionaries.
        # Every dict has the actions as key and another dict as value.
        # The second dict has 'state', 'new_state', 'abs_reward', 'abs_tf' as keys.
        self.container = self.init_container()

        for sam in samples:
            for s in sam:
                # every s is an array with this shape: ['state', 'action', 'reward', 'new_state']
                mcrst = helper.get_mcrst(s[0], self.intervals)
                if problem == 'lqg1d':
                    abs_rew = helper.calc_abs_reward(self.intervals, mcrst, s[1])
                elif problem == 'cartpole1d':
                    abs_rew = 1
                self.container[mcrst][s[1]] = {'state': s[0], 'new_state': s[3], 'abs_reward': abs_rew}

        # to avoid a slow computation.
        self.container = [helper.big_mcrst_correction(cont) if len(cont.items()) > helper.MAX_SAMPLES_IN_MCRST else cont
                          for cont in self.container]

    def compute_abstract_tf(self):
        for cont in self.container:
            for act in cont.keys():
                cont[act]['abs_tf'] = self.calculate_single_atf(cont, act)

    def calculate_single_atf(self, cont, act):
        pass

    # probabilities is a matrix with the shape (#actions, #mcrst)
    def set_abstract_tf(self, probabilities, id_actions):
        for i in range(0, len(self.container)):
            for act in self.container[i].keys():
                id_act = id_actions[act]
                self.container[i][act]['abs_tf'] = probabilities[id_act]
