import numpy as np
from DPO.algorithm.abstraction.abstraction import Abstraction


class LipschitzAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)

    def compute_abstract_tf(self, optA, std=0):
        range_max = len(self.container) if not self.sink else len(self.container) - 1
        for i in range(0, range_max):
            for act in self.container[i].keys():
                self.container[i][act]['abs_tf'] = self.calculate_single_atf(i, act, optA, std)

        if self.sink:
            # sink_tf is the tf array associated to actions in sink state
            sink_tf = np.zeros(len(self.intervals) + 1)
            sink_tf[-1] = 1
            for act in self.container[-1].keys():
                self.container[-1][act]['abs_tf'] = sink_tf

    def calculate_single_atf(self, cont, act, optA):
        pass
