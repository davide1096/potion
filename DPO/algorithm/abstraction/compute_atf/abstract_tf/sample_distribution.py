import numpy as np
import DPO.helper as helper


def abstract_tf(intervals, new_state_bounds, sink):

    adder = 1 if sink else 0
    shape = [len(i) + adder for i in intervals]
    abs_tf = np.zeros(tuple(shape))

    for ns in new_state_bounds:
        min_mcrst = helper.get_mcrst(ns[0], intervals, sink)
        max_mcrst = helper.get_mcrst(ns[1], intervals, sink)

        if min_mcrst == max_mcrst:  # best case: min and max are the same macrostate.
            abs_tf[tuple(min_mcrst)] += 1

        else:
            den = ns[1] - ns[0]
            multi_dim_prob = []

            for i, inter in enumerate(intervals):  # I calculate the probability for every dim separately.
                single_dim_prob = np.zeros(len(inter) + adder)
                if min_mcrst[i] == -1:
                    single_dim_prob[min_mcrst[i]] += (inter[0][0] - ns[0][i]) / den[i]
                else:
                    single_dim_prob[min_mcrst[i]] += (inter[min_mcrst[i]][1] - ns[0][i]) / den[i]
                if max_mcrst[i] == len(inter):
                    single_dim_prob[max_mcrst[i]] += (ns[1][i] - inter[-1][1]) / den[i]
                else:
                    single_dim_prob[max_mcrst[i]] += (ns[1][i] - inter[max_mcrst[i]][0]) / den[i]

                for i in range(min_mcrst[i] + 1, max_mcrst[i]):
                    single_dim_prob[i] += (inter[i][1] - inter[i][0]) / den[i]
                multi_dim_prob.append(single_dim_prob)

            atf_to_sum = np.ones(tuple(shape))
            for axis, multi in enumerate(multi_dim_prob):  # I calculate the probability for every mcrst by combining probabilities on each dimension.

                # I multiply the probability of the matrix atf_to_sum by the probability derived from each dimension.
                dim_array = np.ones((1, atf_to_sum.ndim), int).ravel()
                dim_array[axis] = -1
                b_reshaped = multi.reshape(dim_array)
                atf_to_sum = atf_to_sum * b_reshaped

            abs_tf += helper.normalize_array(atf_to_sum)

    return helper.normalize_array(abs_tf)
