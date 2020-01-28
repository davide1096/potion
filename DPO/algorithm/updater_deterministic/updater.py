import random
import DPO.helper as helper
import numpy as np

# LR_DET_POLICY = 0.025  # 0.05 - 0.01 - 0.005
# N_ITERATIONS_BATCH_GRAD = 200
# BATCH_SIZE = 50
# LAMBDA = 0.0001  # 0.0001


class Updater(object):

    def __init__(self, seed=None, alpha=0.025, lam=0.0005):
        super().__init__()
        self.alpha = alpha
        self.lam = lam
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def batch_gradient_update(self, det_param, samples):
        init_par = det_param
        samples = helper.flat_listoflists(samples)
        samples = random.sample(samples, len(samples))

        # Batch gradient update (previous implementation).
        # for e in range(0, N_ITERATIONS_BATCH_GRAD):
        #     accumulator = np.zeros_like(det_param)
        #     for b in range(0, BATCH_SIZE):
        #         s = samples[random.randint(0, len(samples) - 1)]
        #         accumulator += (np.dot(det_param, s[0]) - s[1]) * s[0]
        #     det_param = det_param - LR_DET_POLICY * (accumulator / BATCH_SIZE + LAMBDA * np.sign(det_param - init_par))

        for s in samples:
            grad = (np.dot(det_param, s[0]) - s[1]) * s[0]
            det_param = det_param - self.alpha * (grad + self.lam * np.sign(det_param - init_par))
        return det_param

    # def calculate_error(self, det_param, samples):
    #     err = 0
    #     for s in samples:
    #         a = np.dot(det_param, s[0]) - s[1]
    #         b = a ** 2
    #         c = 0.5 * b
    #         err += 0.5 * (np.dot(det_param, s[0]) - s[1]) ** 2
    #     return err