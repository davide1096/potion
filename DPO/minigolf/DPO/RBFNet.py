import numpy as np
import matplotlib.pyplot as plt
import random

BATCH_SIZE = 50
LAMBDA = 0.005


def rbf(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, centers, sigma, w, seed, k=5, lr=0.01, epochs=200, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        self.centers = centers
        self.stds = np.repeat(sigma, k)
        self.w = w

        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def fit(self, X, y):

        old_w = self.w.copy()

        new_w = self.w.copy()

        for epoch in range(self.epochs):
            w_accumulator = 0
            for i in range(0, BATCH_SIZE):
                rand_num = random.randint(0, len(X) - 1)
                sample = X[rand_num]

                # forward pass
                a = np.array([self.rbf(sample, c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(new_w)

                # backward pass
                error = -(y[rand_num] - F).flatten()

                # online update
                w_accumulator += a * error

            reg = np.sign([neww - oldw for neww, oldw in zip(new_w, old_w)])  # L1-norm
            # reg = np.array([neww - oldw for neww, oldw in zip(new_w, old_w)])  # L2-norm
            new_w = new_w - self.lr * (w_accumulator / BATCH_SIZE + LAMBDA * reg)

        # self.w = [ALFA * w_old + (1 - ALFA) * w_new for w_old, w_new in zip(self.w, new_w)]
        self.w = new_w

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = np.array([a.T.dot(self.w)])
            y_pred.append(F)
        return np.array(y_pred)
