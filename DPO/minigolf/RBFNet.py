import numpy as np
import matplotlib.pyplot as plt
import random

BATCH_SIZE = 50
ALFA = 0


def rbf(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, centers, w, b, seed, k=5, lr=0.01, epochs=200, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        self.centers = centers
        self.stds = np.repeat(3, k)

        self.w = w
        self.b = np.random.randn(1)
        self.b[0] = b

        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def fit(self, X, y):

        new_w = self.w.copy()
        new_b = self.b.copy()

        for epoch in range(self.epochs):
            w_accumulator = 0
            b_accumulator = 0
            for i in range(0, BATCH_SIZE):
                rand_num = random.randint(0, len(X) - 1)
                sample = X[rand_num]

                # forward pass
                a = np.array([self.rbf(sample, c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(new_w) + new_b

                # backward pass
                error = -(y[rand_num] - F).flatten()

                # online update
                w_accumulator += a * error
                b_accumulator += error

            new_w = new_w - self.lr * (w_accumulator / BATCH_SIZE)
            new_b = new_b - self.lr * (b_accumulator / BATCH_SIZE)

        print(new_w * 0.5)

        self.w = [ALFA * w_old + (1 - ALFA) * w_new for w_old, w_new in zip(self.w, new_w)]
        self.b = [ALFA * b_old + (1 - ALFA) * b_new for b_old, b_new in zip(self.b, new_b)]

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)
