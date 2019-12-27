import random
import DPO.helper as helper
import numpy as np

LR_DET_POLICY = 0.01
N_ITERATIONS_BATCH_GRAD = 200
BATCH_SIZE = 50
LAMBDA = 0.005


class Updater(object):

    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def batch_gradient_update(self, det_param, samples):
        init_par = det_param
        samples = helper.flat_listoflists(samples)
        for e in range(0, N_ITERATIONS_BATCH_GRAD):
            accumulator = np.zeros_like(det_param)
            for b in range(0, BATCH_SIZE):
                s = samples[random.randint(0, len(samples) - 1)]
                accumulator += (np.dot(det_param, s[0]) - s[1]) * s[0]
            det_param = det_param - LR_DET_POLICY * (accumulator / BATCH_SIZE + LAMBDA * np.sign(det_param - init_par))
        return det_param
