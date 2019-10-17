import random
import lqg1Dscalable.helper as helper

random.seed(helper.SEED)

LR_DET_POLICY = 0.1
N_ITERATIONS_BATCH_GRAD = 200
BATCH_SIZE = 50


def batch_gradient_update(det_param, samples):
    for e in range(0, N_ITERATIONS_BATCH_GRAD):
        accumulator = 0
        for b in range(0, BATCH_SIZE):
            s = samples[random.randint(0, len(samples) - 1)]
            accumulator += (det_param * s[0] - s[1]) * s[0]
        det_param = det_param - LR_DET_POLICY * (accumulator / BATCH_SIZE)
    return det_param
