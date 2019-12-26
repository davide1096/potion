import numpy as np
from DPO.algorithm.abstraction.abstraction import Abstraction


class LipschitzAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals=None, Q=None, R=None):
        super().__init__(gamma, sink, intervals, Q, R)

    def compute_abstract_tf(self, optA, std=0):
        for i in range(len(self.container)):
            for act in self.container[i].keys():
                self.container[i][act]['abs_tf'] = self.calculate_single_atf(i, act, optA, std)

    def calculate_single_atf(self, cont, act, optA):
        pass
