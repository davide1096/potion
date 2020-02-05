import random
import DPO.helper as helper
import numpy as np

LR_DET_POLICY = 0.005  # 0.05 - 0.01 - 0.005
# N_ITERATIONS_BATCH_GRAD = 200
# BATCH_SIZE = 50
# LAMBDA = 0.01  # 0.0001


class Updater(object):

    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def gradient_update(self, det_param, samples, lam):
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
            det_param = det_param - LR_DET_POLICY * (grad + lam * np.sign(det_param - init_par))
        return det_param

    # def calculate_error(self, det_param, samples):
    #     err = 0
    #     for s in samples:
    #         a = np.dot(det_param, s[0]) - s[1]
    #         b = a ** 2
    #         c = 0.5 * b
    #         err += 0.5 * (np.dot(det_param, s[0]) - s[1]) ** 2
    #     return err