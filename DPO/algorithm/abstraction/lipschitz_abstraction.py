import numpy as np
from DPO.algorithm.abstraction.abstraction import Abstraction


class LipschitzAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)

    def compute_abstract_tf(self, optA, est_ds, ldeltas=0, models=None):
        range_max = len(self.container) if not self.sink else len(self.container) - 1
        for i in range(0, range_max):
            if est_ds:
                model = None if i==0 else models[i-1]
            for act in self.container[i].keys():
                self.container[i][act]['abs_tf'] = self.calculate_single_atf(i, act, optA, est_ds, ldeltas,
                                                                             model if est_ds else None)

        if self.sink:
            # sink_tf is the tf array associated to actions in sink state
            if ldeltas == 0:
                sink_tf = np.zeros(len(self.intervals) + 1)
            else:
                prob = [0., 0.]
                sink_tf = [prob for i in range(len(self.intervals) + 1)]
            # sink_tf[-1] = 1
            for act in self.container[-1].keys():
                self.container[-1][act]['abs_tf'] = sink_tf

    def calculate_single_atf(self, mcrst, act, ds0, est_ds, ldeltas=0, model=None):
        pass
