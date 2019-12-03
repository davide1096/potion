import random
import lqg1Dscalable.helper as helper
import numpy as np

LR_DET_POLICY = 0.1
N_ITERATIONS_BATCH_GRAD = 200
BATCH_SIZE = 50


class Updater(object):

    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def batch_gradient_update(self, det_param, samples):
        samples = helper.flat_listoflists(samples)
        for e in range(0, N_ITERATIONS_BATCH_GRAD):
            accumulator = 0
            for b in range(0, BATCH_SIZE):
                s = samples[random.randint(0, len(samples) - 1)]
                accumulator += (det_param * s[0] - s[1]) * s[0]
            det_param = det_param - LR_DET_POLICY * (accumulator / BATCH_SIZE)
        return det_param


# gradient update functions used in minigolf task

# def batch_gradient_update_a(a_par, b_par, samples):
#     samples = helper.flat_listoflists(samples)
#     for e in range(0, N_ITERATIONS_BATCH_GRAD):
#         accumulator = 0
#         for b in range(0, BATCH_SIZE):
#             s = samples[random.randint(0, len(samples) - 1)]
#             accumulator += (np.sqrt(a_par * s[0] + b_par) - s[1]) * (0.5 / np.sqrt(a_par * s[0] + b_par)) * s[0]
#         a_par = a_par - LR_DET_POLICY * (accumulator / BATCH_SIZE)
#     return a_par
#
#
# def batch_gradient_update_b(a_par, b_par, samples):
#     samples = helper.flat_listoflists(samples)
#     for e in range(0, N_ITERATIONS_BATCH_GRAD):
#         accumulator = 0
#         for b in range(0, BATCH_SIZE):
#             s = samples[random.randint(0, len(samples) - 1)]
#             accumulator += (np.sqrt(a_par * s[0] + b_par) - s[1]) * (0.5 / np.sqrt(a_par * s[0] + b_par))
#         b_par = b_par - LR_DET_POLICY * (accumulator / BATCH_SIZE)
#     return b_par