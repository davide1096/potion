import numpy as np
import matplotlib.pyplot as plt
import random
import lqg1Dscalable.helper as helper

random.seed(helper.SEED)
BATCH_SIZE = 50


def rbf(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=5, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        # ---
        self.centers = [3, 6, 10, 14, 17]
        self.stds = np.repeat(3, k)
        # ---

        self.w = np.random.randn(k)
        for i in range(len(self.w)):
            self.w[i] = 0.5
        self.b = np.random.randn(1)
        for i in range(len(self.b)):
            self.b[i] = 1

    def fit(self, X, y):

        # training
        # for epoch in range(self.epochs):
        #     for i in range(X.shape[0]):
        #         # forward pass
        #         a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
        #         F = a.T.dot(self.w) + self.b
        #
        #         loss = (y[i] - F).flatten() ** 2
        #         # print('Loss: {0:.2f}'.format(loss[0]))
        #
        #         # backward pass
        #         error = -(y[i] - F).flatten()
        #
        #         # online update
        #         self.w = self.w - self.lr * a * error
        #         self.b = self.b - self.lr * error

        for epoch in range(self.epochs):
            w_accumulator = 0
            b_accumulator = 0
            for i in range(0, BATCH_SIZE):
                rand_num = random.randint(0, len(X) - 1)
                sample = X[rand_num]

                # forward pass
                a = np.array([self.rbf(sample, c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                # backward pass
                error = -(y[rand_num] - F).flatten()

                # online update
                w_accumulator += a * error
                b_accumulator += error

            self.w = self.w - self.lr * (w_accumulator / BATCH_SIZE)
            self.b = self.b - self.lr * (b_accumulator / BATCH_SIZE)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)
