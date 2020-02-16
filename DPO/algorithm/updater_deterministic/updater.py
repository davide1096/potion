import random
import numpy as np


class Updater(object):

    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def gradient_update(self, det_param, samples, alpha, lam):
        init_par = det_param
        samples = random.sample(samples, len(samples))
        der_base = np.zeros((len(det_param),))
        for s in samples:
            grad = np.empty_like(det_param)
            for i in range(len(det_param)):
                for j in range(len(det_param[i])):
                    der = der_base
                    der[i] = s[0][j]
                    grad[i][j] = np.dot((np.dot(det_param, s[0]) - s[1]), der)
            det_param = det_param - alpha * (grad + lam * np.sign(det_param - init_par))
        return det_param
