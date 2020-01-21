import cvxpy as cp
import numpy as np
from DPO.algorithm.abstraction.abstraction import Abstraction
import DPO.helper as helper
import DPO.algorithm.abstraction.helper_maxlikelihood as helper_maxlikelihood


class MaxLikelihoodAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals, L, Q=None, R=None):
        super().__init__(gamma, sink, intervals, Q, R)
        self.i = None
        self.L = L
        self.n_actions = None
        self.I = None
        self.action_index = None
        self.arriving_mcrst_helper = None
        self.solution = None

    def init_operation(self):
        self.i = len(self.container)  # Number of columns of the matrix.
        self.n_actions = self.count_actions()  # Number of rows of the matrix.
        self.I = cp.Parameter((self.n_actions, self.i), nonneg=True)  # Matrix that represents the abstract TPs.
        self.action_index = {}  # Index of every action in the I matrix.
        self.create_action_index()
        self.arriving_mcrst_helper = {}  # It allows to consider fictitious samples.
        self.create_arriving_mcrst_helper()
        self.fill_I()

    # For every macrostate I count the number of performed actions.
    def count_actions(self):
        return sum([len(cont.items()) for cont in self.container])

    def create_action_index(self):
        id = 0
        for i in range(0, self.i):
            for act in self.container[i].keys():
                if act not in self.action_index:
                    self.action_index[act] = id
                    id += 1
        assert (id == self.n_actions)  # Ensure to not have the same action with two different indexes.

    def get_id_from_action(self, action):
        return self.action_index[action]

    # def get_action_from_id(self, id):
    #     return [k for k, v in self.action_index.items() if v == id][0]

    def fill_I(self):

        matrix_i = np.zeros((self.n_actions, self.i))
        for cont in self.container:

            for act, single_sample in cont.items():
                new_mcrst = helper.get_mcrst(single_sample['new_state'], self.intervals, self.sink)
                # I assume that all the actions are different.
                matrix_i[self.get_id_from_action(act)][helper.get_index_from_mcrst(new_mcrst, self.intervals)] += 1

            # contribution of the fictitious samples.
            for act in self.arriving_mcrst_helper.keys():
                for mcrst in self.arriving_mcrst_helper[act].keys():
                    matrix_i[self.get_id_from_action(act)][mcrst] += self.arriving_mcrst_helper[act][mcrst]

        self.I.value = matrix_i

    def create_arriving_mcrst_helper(self):

        for cont in self.container:
            for act in cont.keys():

                # Evaluate the effect of act on every sample in the macrostate.
                # --> We assume valid the Lipschitz-0 hypothesis on delta s in order to add fictitious samples! <--
                sample = cont[act]
                delta_s = sample['new_state'] - sample['state']
                self.arriving_mcrst_helper[act] = {}

                # Apply the delta s of the sample to every other state in the macrostate.
                for act2 in cont.keys():
                    if act != act2:
                        new_state = cont[act2]['state'] + delta_s
                        new_state_mcrst = helper.get_mcrst(new_state, self.intervals, self.sink)
                        index = helper.get_index_from_mcrst(new_state_mcrst, self.intervals)

                        if index in self.arriving_mcrst_helper[act].keys():
                            self.arriving_mcrst_helper[act][index] += 1
                        else:
                            self.arriving_mcrst_helper[act][index] = 1

    def construct_problem(self):
        self.init_operation()  # Initialize some variables of support.
        theta = cp.Variable((self.n_actions, self.i), nonneg=True)
        objective = cp.Minimize(-cp.sum(cp.multiply(self.I, cp.log(theta))))

        constraints = []
        # Sum of rows must be equal to 1.
        for k in range(0, self.n_actions):
            constraints.append(cp.sum(theta[k]) == 1)

        lipschitz_cons = helper_maxlikelihood.compute_lipschitz_constraints(self.container, self.intervals, self.sink,
                                                                            self.arriving_mcrst_helper,
                                                                            self.action_index, theta, self.L)

        problem = cp.Problem(objective, constraints + lipschitz_cons)
        problem.solve(solver=cp.ECOS, abstol=1e-4, max_iters=200)

        return theta.value

    def get_abstract_tf(self):
        return self.solution, self.action_index

    def compute_abstract_tf(self, optA, mins=-1, maxs=1, maxa=1, std=0):
        self.solution = self.construct_problem()  # Compute the abstract transition function for every action.
        shape = [len(i) for i in self.intervals]
        for i in range(0, len(self.container)):
            for act in self.container[i].keys():
                id_act = self.action_index[act]
                self.container[i][act]['abs_tf'] = np.array(self.solution[id_act]).reshape(tuple(shape))

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
