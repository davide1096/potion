import numpy as np
from DPO.algorithm.abstraction.abstraction import Abstraction


class LipschitzAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals=None, Q=None, R=None, maxa_env=1):
        super().__init__(gamma, sink, intervals, Q, R, maxa_env)

    def compute_abstract_tf(self, optA):
        for i in range(len(self.container)):
            for k in self.container[i].keys():
                self.container[i][k]['abs_tf'] = self.calculate_single_atf(i, k, optA)

    def calculate_single_atf(self, cont, act, optA):
        pass
