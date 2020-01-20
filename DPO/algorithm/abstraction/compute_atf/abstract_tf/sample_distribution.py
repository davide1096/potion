import numpy as np
import DPO.helper as helper


def abstract_tf(intervals, new_states, sink):

    adder = 1 if sink else 0
    shape = [len(i) + adder for i in intervals]
    abs_tf = np.zeros(tuple(shape))

    for ns in new_states:
        ns_mcrst = helper.get_mcrst(ns, intervals, sink)
        abs_tf[tuple(ns_mcrst)] += 1

    return helper.normalize_array(abs_tf)
