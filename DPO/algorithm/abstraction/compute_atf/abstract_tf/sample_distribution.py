import numpy as np
import DPO.helper as helper


def abstract_tf(intervals, new_state_bounds, sink):

    adder = 1 if sink else 0
    abs_tf = np.zeros(len(intervals) + adder)
    for ns in new_state_bounds:
        min_mcrst = helper.get_mcrst(ns[0], intervals, sink)
        max_mcrst = helper.get_mcrst(ns[1], intervals, sink)

        if min_mcrst == max_mcrst:
            abs_tf[min_mcrst] += 1

        else:
            den = ns[1] - ns[0]

            if min_mcrst == -1:
                abs_tf[min_mcrst] += (intervals[0][0] - ns[0]) / den
            else:
                abs_tf[min_mcrst] += (intervals[min_mcrst][1] - ns[0]) / den
            if max_mcrst == len(intervals):
                abs_tf[max_mcrst] += (ns[1] - intervals[-1][1]) / den
            else:
                abs_tf[max_mcrst] += (ns[1] - intervals[max_mcrst][0]) / den

            for i in range(min_mcrst + 1, max_mcrst):
                abs_tf[i] += (intervals[i][1] - intervals[i][0]) / den

    return helper.normalize_array(abs_tf)
