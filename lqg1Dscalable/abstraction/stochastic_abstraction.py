import cvxpy as cp
import numpy as np
import lqg1Dv2.abstraction as ab
from lqg1Dscalable.abstraction.abstraction import Abstraction


class StochasticAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals, L):
        super().__init__(gamma, sink, intervals)
        self.i = None
        self.L = L
        self.n_actions = None
        self.I = None
        self.action_index = {}
        self.solution = None

    def init_operation(self):
        self.i = len(self.container)
        self.n_actions = self.count_actions()
        self.I = cp.Parameter((self.n_actions, self.i))
        self.action_index = {}
        self.create_action_index()
        self.fill_I()

    # for every macrostate I count the number of actions performed in the samples.
    def count_actions(self):
        return sum([len(cont.items()) for cont in self.container])

    def create_action_index(self):
        id = 0
        for i in range(0, self.i):
            for act in self.container[i].keys():
                if act not in self.action_index:
                    self.action_index[act] = id
                    id += 1

    def get_id_from_action(self, action):
        return self.action_index[action]

    def get_action_from_id(self, id):
        return [k for k, v in self.action_index.items() if v == id][0]

    def fill_I(self):
        matrix_i = np.zeros((self.n_actions, self.i))
        for i in range(0, self.i):
            for act in self.container[i].keys():
                single_sample = self.container[i][act]
                new_mcrst = ab.get_mcrst(single_sample['new_state'], self.intervals)
                # I assume that all the actions are different.
                matrix_i[self.get_id_from_action(act)][new_mcrst] += 1
        self.I.value = matrix_i

    def construct_problem(self):
        self.init_operation()
        theta = cp.Variable((self.n_actions, self.i), nonneg=True)
        objective = cp.Minimize(-cp.sum(cp.log(cp.multiply(self.I, theta) + 1)))

        constraints = []
        # sum of rows = 1
        for k in range(0, self.n_actions):
            constraints.append(cp.sum(theta[k]) == 1)

        # Lipschitz hypothesis between actions in the same macrostate
        for k in range(0, self.i):
            actions_mcrst = sorted(list(self.container[k].keys()))
            for i in range(0, len(actions_mcrst) - 1):
                for k2 in range(0, self.i):
                    constraints.append(theta[self.get_id_from_action(actions_mcrst[i])][k2] -
                                       theta[self.get_id_from_action(actions_mcrst[i+1])][k2] <=
                                       self.L * abs(actions_mcrst[i] - actions_mcrst[i+1]))
                    constraints.append(theta[self.get_id_from_action(actions_mcrst[i])][k2] -
                                       theta[self.get_id_from_action(actions_mcrst[i+1])][k2] >=
                                       - self.L * abs(actions_mcrst[i] - actions_mcrst[i+1]))

        problem = cp.Problem(objective, constraints)
        problem.solve()
        print(theta.value)

        return theta.value

    def get_abstract_tf(self):
        return self.solution, self.action_index

    def compute_abstract_tf(self):
        self.solution = self.construct_problem()
        for i in range(0, len(self.container)):
            for act in self.container[i].keys():
                id_act = self.action_index[act]
                self.container[i][act]['abs_tf'] = self.solution[id_act]

        if self.sink:
            sink_tf = np.zeros(len(self.intervals) + 1)
            sink_tf[-1] = 1
            for act in self.container[-1].keys():
                self.container[-1][act]['abs_tf'] = sink_tf


# test

# lip = 0.1
# intervals = [[0.0, 0.2], [0.2, 0.4]]
# samples = []
# for i in range(0, len(intervals)):
#     samples.append({})
# samples[0][0] = [None, None, 0.1, 0.1]
# samples[0][2] = [None, None, 0.1, 0.3]
# samples[0][1] = [None, None, 0.1, 0.1]
# samples[0][3] = [None, None, 0.1, 0.3]
# samples[1][4] = [None, None, 0.3, 0.3]
# tester = AbstractTF(samples, lip, intervals)