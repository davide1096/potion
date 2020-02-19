import numpy as np
import DPO.helper as helper


# given a pair (x, a) it returns an array representing min and max p(x'|x,a) for each x'
def abstract_tf(intervals, new_state_bounds, sink):

    adder = 1 if sink else 0
    shape = [len(i) + adder for i in intervals]
    abs_tf_min = np.zeros(tuple(shape))
    abs_tf_max = np.zeros(tuple(shape))

    for ns in new_state_bounds:
        min_mcrst = helper.get_mcrst(ns[0], intervals, sink)
        max_mcrst = helper.get_mcrst(ns[1], intervals, sink)

        # update min interval.
        if min_mcrst == max_mcrst:
            abs_tf_min[tuple(min_mcrst)] += 1

        # update max interval.
        multi_dim_prob = []
        for dim in range(len(intervals)):
            single_dim_prob = np.zeros(len(intervals[dim]) + adder)
            for i in range(min_mcrst[dim], max_mcrst[dim] + 1):
                single_dim_prob[i] += 1

            # correction (ev).
            if min_mcrst[dim] == -1 and max_mcrst[dim] == len(intervals[dim]):
                single_dim_prob[min_mcrst] -= 1
            multi_dim_prob.append(single_dim_prob)

        atf_to_sum = np.ones(tuple(shape))
        for axis, multi in enumerate(multi_dim_prob):  # I calculate the probability for every mcrst by combining probabilities on each dimension.

            # I multiply the probability of the matrix atf_to_sum by the probability derived from each dimension.
            dim_array = np.ones((1, atf_to_sum.ndim), int).ravel()
            dim_array[axis] = -1
            b_reshaped = multi.reshape(dim_array)
            atf_to_sum = atf_to_sum * b_reshaped

        abs_tf_max += atf_to_sum

    # normalization.
    den = len(new_state_bounds)
    return np.stack((abs_tf_min/den, abs_tf_max/den), axis=0)

