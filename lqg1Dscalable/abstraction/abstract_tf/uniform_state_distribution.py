import numpy as np
import lqg1Dscalable.helper as helper


def abstract_tf(intervals, new_state_bounds, sink):

    adder = 1 if sink else 0
    abs_tf = np.zeros(len(intervals) + adder)
    # I obtain the min & max new state I would get by performing action act in the mcrst, according to the samples.
    new_st_min = min([ns[0] for ns in new_state_bounds])
    new_st_max = max([ns[1] for ns in new_state_bounds])

    min_mcrst = helper.get_mcrst(new_st_min, intervals, sink)
    max_mcrst = helper.get_mcrst(new_st_max, intervals, sink)

    if min_mcrst == max_mcrst:
        abs_tf[min_mcrst] += 1

    else:
        if min_mcrst == -1:
            abs_tf[min_mcrst] += (intervals[0][0] - new_st_min)
        else:
            abs_tf[min_mcrst] += (intervals[min_mcrst][1] - new_st_min)
        if max_mcrst == len(intervals):
            abs_tf[max_mcrst] += (new_st_max - intervals[-1][1])
        else:
            abs_tf[max_mcrst] += (new_st_max - intervals[max_mcrst][0])

        for i in range(min_mcrst + 1, max_mcrst):
            abs_tf[i] += (intervals[i][1] - intervals[i][0])

    return helper.normalize_array(abs_tf)
