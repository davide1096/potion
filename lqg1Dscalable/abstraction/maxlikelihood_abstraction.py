import cvxpy as cp
import numpy as np
from lqg1Dscalable.abstraction.abstraction import Abstraction
import lqg1Dscalable.helper as helper


class MaxLikelihoodAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals, L):
        super().__init__(gamma, sink, intervals)
        self.i = None
        self.L = L
        # n_actions is the number of row of the matrix.
        self.n_actions = None
        self.I = None
        self.action_index = {}
        # it allows to consider fictitious samples.
        self.arriving_mcrst_helper = {}
        self.solution = None

    def init_operation(self):
        self.i = len(self.container)
        self.n_actions = self.count_actions()
        self.I = cp.Parameter((self.n_actions, self.i))
        self.action_index = {}
        self.create_action_index()
        self.arriving_mcrst_helper = {}
        self.create_arriving_mcrst_helper()
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
        assert (id == self.n_actions)

    def get_id_from_action(self, action):
        return self.action_index[action]

    # def get_action_from_id(self, id):
    #     return [k for k, v in self.action_index.items() if v == id][0]

    def fill_I(self):

        matrix_i = np.zeros((self.n_actions, self.i))
        for i in range(0, self.i):

            for act in self.container[i].keys():
                single_sample = self.container[i][act]
                new_mcrst = helper.get_mcrst(single_sample['new_state'], self.intervals, self.sink)
                # I assume that all the actions are different.
                matrix_i[self.get_id_from_action(act)][new_mcrst] += 1

            # contribution of the fictitious samples.
            for act in self.arriving_mcrst_helper.keys():
                for mcrst in self.arriving_mcrst_helper[act].keys():
                    matrix_i[self.get_id_from_action(act)][mcrst] += self.arriving_mcrst_helper[act][mcrst]

        self.I.value = matrix_i

    def create_arriving_mcrst_helper(self):

        for cont in self.container:
            for act in cont.keys():

                # evaluate the effect of act on the single sample.
                sample = cont[act]
                delta_s = sample['new_state'] - sample['state']
                self.arriving_mcrst_helper[act] = {}

                # apply the effect to every state in the macrostate.
                for act2 in cont.keys():
                    if act != act2:
                        new_state = cont[act2]['state'] + delta_s
                        new_state_mcrst = helper.get_mcrst(new_state, self.intervals, self.sink)

                        if new_state_mcrst in self.arriving_mcrst_helper[act].keys():
                            self.arriving_mcrst_helper[act][new_state_mcrst] += 1
                        else:
                            self.arriving_mcrst_helper[act][new_state_mcrst] = 1

    def construct_problem(self):
        self.init_operation()
        theta = cp.Variable((self.n_actions, self.i), nonneg=True)
        objective = cp.Minimize(-cp.sum(cp.log(cp.multiply(self.I, theta) + 1)))

        constraints = []
        # sum of rows must be equal to 1.
        for k in range(0, self.n_actions):
            constraints.append(cp.sum(theta[k]) == 1)

        # Lipschitz hypothesis between actions in the same macrostate.
        for k in range(0, self.i):

            actions_mcrst = sorted(list(self.container[k].keys()), reverse=True)
            new_mcrst_possible = []
            for act in actions_mcrst:
                new_mcrst = helper.get_mcrst(self.container[k][act]['new_state'], self.intervals, self.sink)

                if new_mcrst not in new_mcrst_possible:
                    new_mcrst_possible.append(new_mcrst)

                # from helper might contain new_mcrst that are not yet included in new_mcrst_possible.
                from_helper = self.arriving_mcrst_helper[act].keys()
                for mcrst in from_helper:
                    if mcrst not in new_mcrst_possible:
                        new_mcrst_possible.append(mcrst)

            for i in range(0, len(actions_mcrst) - 1):
                for k2 in new_mcrst_possible:
                    constraints.append(theta[self.get_id_from_action(actions_mcrst[i])][k2] -
                                       theta[self.get_id_from_action(actions_mcrst[i + 1])][k2] <=
                                       self.L * abs(actions_mcrst[i] - actions_mcrst[i + 1]))
                    constraints.append(theta[self.get_id_from_action(actions_mcrst[i])][k2] -
                                       theta[self.get_id_from_action(actions_mcrst[i + 1])][k2] >=
                                       - self.L * abs(actions_mcrst[i] - actions_mcrst[i + 1]))

        problem = cp.Problem(objective, constraints)
        problem.solve()
        print(theta.value)

        return theta.value

    def get_abstract_tf(self):
        return self.solution, self.action_index

    def compute_abstract_tf(self, std=0):
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
