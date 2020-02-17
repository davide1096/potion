import numpy as np
from DPO.algorithm.abstraction.abstraction import Abstraction


class LipschitzAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals=None):
        super().__init__(gamma, sink, intervals)

    def compute_abstract_tf(self, Lds=0):
        for k1, v1 in self.container.items():
            for k2, v2 in v1.items():
                v2['abs_tf'] = self.calculate_single_atf(k1, k2, Lds)

    def calculate_single_atf(self, cont, act, Lds):
        pass
