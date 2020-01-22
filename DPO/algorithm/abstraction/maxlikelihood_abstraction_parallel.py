import cvxpy as cp
import numpy as np
from DPO.algorithm.abstraction.abstraction import Abstraction
import DPO.helper as helper
import DPO.algorithm.abstraction.helper_maxlikelihood as helper_maxlikelihood
import multiprocessing as mp
import os


class MaxLikelihoodAbstraction(Abstraction):

    def __init__(self, gamma, sink, intervals, L, Q=None, R=None):
        super().__init__(gamma, sink, intervals, Q, R)
        self.i = None
        self.L = L
        self.arriving_mcrst_helper = {}
        self.results = []

    def calculate_arriving_mcrst_list(self, mcrst):
        arriving_mcrst = []
        actions = self.container[mcrst].keys()

        for act in actions:
            from_helper = self.arriving_mcrst_helper[act].keys()
            for index_mcrst in from_helper:
                if index_mcrst not in arriving_mcrst:
                    arriving_mcrst.append(index_mcrst)

        return arriving_mcrst

    def fill_I(self, mcrst, ordered_actions):

        matrix_i = np.zeros((len(ordered_actions), self.i))
        cont = self.container[mcrst]

        for i, act in enumerate(ordered_actions):
            for index_mcrst in self.arriving_mcrst_helper[act].keys():
                matrix_i[i][index_mcrst] += self.arriving_mcrst_helper[act][index_mcrst]

        return matrix_i

    def create_arriving_mcrst_helper(self):

        for cont in self.container:
            for act in cont.keys():

                # Evaluate the effect of act on every sample in the macrostate.
                # --> We assume valid the Lipschitz-0 hypothesis on delta s in order to add fictitious samples! <--
                sample = cont[act]
                delta_s = sample['new_state'] - sample['state']
                self.arriving_mcrst_helper[act] = {}  # every action is a key.
                ns = helper.get_mcrst(sample['new_state'], self.intervals, self.sink)
                ns_index = helper.get_multidim_mcrst(ns, self.intervals)
                self.arriving_mcrst_helper[act][ns_index] = 1  # every index of an arriving mcrst is a key.

                # Apply the delta s of the sample to every other state in the macrostate.
                for act2 in cont.keys():
                    if act != act2:  # evaluation of act in all the other samples in the mcrst.
                        new_state = cont[act2]['state'] + delta_s
                        new_state_mcrst = helper.get_mcrst(new_state, self.intervals, self.sink)
                        index = helper.get_multidim_mcrst(new_state_mcrst, self.intervals)

                        if index in self.arriving_mcrst_helper[act].keys():
                            self.arriving_mcrst_helper[act][index] += 1
                        else:
                            self.arriving_mcrst_helper[act][index] = 1  # every index of an arriving mcrst is a key.

    # Create the max-likelihood problem for a single mcrst.
    def construct_problem(self, mcrst):
        actions = self.container[mcrst].keys()
        ordered_actions = sorted(actions, reverse=True)
        arriving_mcrst = self.calculate_arriving_mcrst_list(mcrst)  # list of indexes related to the arriving mcrsts.
        I = self.fill_I(mcrst, ordered_actions)  # every row is ordered according to ordered_actions.
        theta = cp.Variable((len(actions), self.i), nonneg=True)
        objective = cp.Minimize(-cp.sum(cp.multiply(I, cp.log(theta))))

        constraints = []
        # Sum of rows must be equal to 1.
        for k in range(len(actions)):
            constraints.append(cp.sum(theta[k]) == 1)

        # Lipschitz hypothesis between actions.
        for i in range(len(actions) - 1):
            for k2 in arriving_mcrst:
                constraints.append(theta[i][k2] - theta[i+1][k2] <= self.L *
                                   abs(ordered_actions[i] - ordered_actions[i+1]))
                constraints.append(theta[i][k2] - theta[i+1][k2] >= - self.L *
                                   abs(ordered_actions[i] - ordered_actions[i+1]))

        problem = cp.Problem(objective, constraints)
        return problem

    def build_initial_solution(self, mcrst):
        cont = self.container[mcrst]
        actions = cont.keys()
        ordered_actions = sorted(actions, reverse=True)
        init_sol = self.fill_I(mcrst, ordered_actions)
        new_init = []
        for i in range(len(init_sol)):
            new_init.append(helper.normalize_array(init_sol[i]))
        return np.array(new_init)

    def compute_parallel_solution(self, mcrst, problem):

        print("Solving the problem for the mcrst: {}".format(mcrst))
        if problem is not None:
            # initial_solution = self.build_initial_solution(i)
            # p.variables()[0] = initial_solution
            try:
                problem.solve(solver=cp.ECOS, max_iters=200)
            except cp.SolverError:
                problem.solve(solver=cp.SCS, max_iters=200)
            # problem.solve(solver=cp.ECOS, max_iters=200, verbose=True)
            theta = problem.variables()[0].value
            return (mcrst, theta)
        else:
            return (mcrst, None)

    def collect_result(self, result):
        self.results.append(result)

    def pre_construct_problem(self, mcrst):
        if len(self.container[mcrst]) > 0:  # I consider not empty macrostate.
            return self.construct_problem(mcrst)
        else:
            return None

    def compute_abstract_tf(self, optA, mins=-1, maxs=1, maxa=1, std=0):
        self.i = len(self.container)  # it represents the # of columns of every matrix.
        self.create_arriving_mcrst_helper()  # it allows to consider fictitious samples.

        # pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool(len(os.sched_getaffinity(0)))  # uses visible cpus
        problems = [pool.apply(self.pre_construct_problem, args=(i, )) for i in [j for j in range(self.i)]]

        self.results = []
        for i, p in enumerate(problems):
            pool.apply_async(self.compute_parallel_solution, args=(i, p), callback=self.collect_result)

        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

        self.results.sort(key=lambda x: x[0])
        solution = [r for i, r in self.results]

        shape = [len(i) for i in self.intervals]
        for mcrst in range(0, len(self.container)):
            actions = self.container[mcrst].keys()
            ordered_actions = sorted(actions, reverse=True)
            for i, act in enumerate(ordered_actions):
                self.container[mcrst][act]['abs_tf'] = np.array(solution[mcrst][i]).reshape(tuple(shape))

        if self.sink:
            sink_tf = np.zeros(len(self.intervals) + 1)
            sink_tf[-1] = 1
            for act in self.container[-1].keys():
                self.container[-1][act]['abs_tf'] = sink_tf

        # Step 3: Don't forget to close
        pool.close()
