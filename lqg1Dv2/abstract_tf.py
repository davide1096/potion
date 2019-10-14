import cvxpy as cp
import numpy as np


class AbstractTF(object):

    def __init__(self, i, j, samples, L):
        super().__init__()
        self.i = i # number of macrostates
        self.j = j # (total) number of actions
        self.samples = samples
        self.L = L
        self.I = cp.Parameter((i*j, i))
        self.action_index = {}
        self.create_action_index()
        self.fill_I()
        self.construct_problem()

    def create_action_index(self):
        id = 0
        for i in range(0, self.i):
            for act in self.samples[i].keys():
                if act not in self.action_index:
                    self.action_index[act] = id
                    id += 1

    def get_id_from_action(self, action):
        return self.action_index[action]

    def get_action_from_id(self, id):
        return [k for k, v in self.action_index.items() if v == id][0]

    def fill_I(self):
        matrix_i = np.zeros((self.i*self.j, self.i))
        for i in range(0, self.i):
            for act in self.samples[i].keys():
                single_sample = self.samples[i][act]
                # for every macrostate I have a block of rows representing all the actions:
                # single_sample[2] * self.j brings me at the beginning of the block
                # self.get_action_id(act) is the offset in the block
                row_index = single_sample[2] * self.j + self.get_id_from_action(act)
                matrix_i[row_index][single_sample[3]] += 1
        self.I.value = matrix_i

    def construct_problem(self):
        theta = cp.Variable((self.i * self.j, self.i), nonneg=True)
        objective = cp.Minimize(-cp.sum(cp.log(cp.multiply(self.I, theta) + 1)))

        constraints = []
        # sum of rows = 1
        for k in range(0, self.i * self.j):
            constraints.append(cp.sum(theta[k]) == 1)

        # Lipschitz hypothesis between actions in the same macrostate
        for k in range(0, self.i):
            for i in range(0, self.j - 1):
                for j in range(i+1, self.j):
                    row_index_i = k * self.j + i
                    row_index_j = k * self.j + j
                    for k2 in range(0, self.i):
                        constraints.append(theta[row_index_i][k2] - theta[row_index_j][k2] <=
                                           self.L * abs(self.get_action_from_id(i) - self.get_action_from_id(j)))
                        constraints.append(theta[row_index_i][k2] - theta[row_index_j][k2] >=
                                           - self.L * abs(self.get_action_from_id(i) - self.get_action_from_id(j)))

        problem = cp.Problem(objective, constraints)
        problem.solve()
        print(theta.value)


# test

n_states = 2
n_actions = 3
lip = 0.1
samples = []
for i in range(0, n_states):
    samples.append({})
samples[0][0] = [None, None, 0, 0]
samples[0][2] = [None, None, 0, 1]
samples[0][1] = [None, None, 0, 0]
samples[1][2] = [None, None, 1, 1]
tester = AbstractTF(n_states, n_actions, samples, lip)