import DPO.helper as helper
from DPO.helper import Helper


class Abstraction(object):

    def __init__(self, gamma, sink, intervals=None, Q=None, R=None, max_a=None):
        super().__init__()
        self.intervals = intervals
        self.container = {}
        self.gamma = gamma
        self.sink = sink
        self.Q = Q
        self.R = R
        self.max_a = max_a

    def get_container(self):
        return self.container

    def divide_samples(self, samples, problem, seed, intervals=None):

        if intervals is not None:
            self.intervals = intervals

        # container is a dict whose key is the macrostate index.
        # For each macrostate a dict contains the samples and the key is an ID.
        # Each sample is a dict having 'state', 'new_state', 'abs_reward', 'abs_tf' as keys.
        self.container = {}
        reward_helper = {}

        for s in samples:
            mcrst = helper.get_mcrst(s[0], self.intervals, self.sink)
            mcrst_index = helper.get_index_from_mcrst(mcrst, self.intervals)
            if mcrst_index != 'sink':  # samples included in the sink are not stored.
                if mcrst_index not in self.container.keys():
                    self.container[mcrst_index] = {}
                    reward_helper[mcrst_index] = {'val': 0, 'den': 0}
                key = len(self.container[mcrst_index].items())
                self.container[mcrst_index][key] = {'state': s[0], 'action': s[1], 'new_state': s[3]}
                reward_helper[mcrst_index]['val'] += s[2]
                reward_helper[mcrst_index]['den'] += 1

        # to avoid a slow computation.
        help = Helper(seed)
        for i in self.container.keys():
            if len(self.container[i].items()) > helper.MAX_SAMPLES_IN_MCRST:
                self.container[i] = help.big_mcrst_correction(self.container[i])

        # compute the abstract reward.
        if problem == "safety":
            for i in self.container.keys():
                abs_rew = reward_helper[i]['val'] / reward_helper[i]['den']
                for _, v in self.container[i].items():
                    v['abs_reward'] = abs_rew
        elif problem == "minigolf":
            for i in self.container.keys():
                for _, v in self.container[i].items():
                    v['abs_reward'] = 0 if i == 0 else -1
        elif problem == "mass":
            for i in self.container.keys():
                for _, v in self.container[i].items():
                    v['abs_reward'] = helper.calc_abs_reward_lqg(self.container[i], v['action'], self.Q, self.R,
                                                                 self.max_a)

    def compute_abstract_tf(self):
        pass

    # the next two functions adapt the representation of the samples.
    # (bounded-MDP and max-likelihood implementation require a different representation).

    def to_old_representation(self):
        container = []
        for i in range(max(self.container.keys())+1):
            container.append({})
        for i in range(max(self.container.keys())+1):
            if i in self.container.keys():
                for id_act, v in self.container[i].items():
                    v['id_action'] = id_act
                    container[i][v['action']] = v
        self.container = container

    def to_new_representation(self, change_tf=True):
        container = {}
        for i in range(len(self.container)):
            if len(self.container[i]) > 0:
                container[i] = {}
                for k, v in self.container[i].items():
                    if change_tf:
                        tf = v['abs_tf']
                        new_tf = {}
                        for i_row, val_row in enumerate(tf):
                            for i_col, val_col in enumerate(tf[i_row]):
                                if val_col > 0:
                                    new_tf[helper.get_index_from_mcrst([i_row, i_col], self.intervals)] = val_col
                        v['abs_tf'] = new_tf
                    container[i][v['id_action']] = v
        self.container = container

