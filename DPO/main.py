import DPO.safetygym.safetygym_DPO as safe_main
import DPO.minigolf.minigolf_DPO as mini_main
import DPO.mass.mass_DPO as mass_main
import multiprocessing as mp
import argparse
import os


def chooser(args):
    task = args['task']
    seed = args['seed']
    runs = args['nruns']

    if task == "mini":
        main_task = mini_main
    elif task == "mass":
        main_task = mass_main
    elif task == "safe":
        main_task = safe_main

    if task == "mass":  # we need to parallelize in the max-likelihood phase.
        for i in range(seed, seed+runs):
            main_task.main(i, args)
    else:
        pool = mp.Pool(len(os.sched_getaffinity(0)))  # uses visible cpus
        for i in range(seed, seed+runs):
            pool.apply_async(main_task.main, args=(i, args))
        pool.close()
        pool.join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Name of the task: mini, mass or safe", type=str)
    ap.add_argument("--seed", required=False, help="Value of the initial seed", default=1, type=int)
    ap.add_argument("--nruns", required=False, help="N. of runs of the task", default=1, type=int)
    ap.add_argument("--alpha", required=False, help="Learning rate used in policy projection", type=float)
    ap.add_argument("--lambda", required=False, help="Regularization used in policy projection", type=float)
    ap.add_argument("--mcrst", required=False, help="Number of abstract states on each state dimension", type=int)
    ap.add_argument("--batch", required=False, help="Size of the batch", type=int)
    ap.add_argument("--nsteps", required=False, help="Size of an episode", type=int)
    ap.add_argument("--niter", required=False, help="Iterations of the algorithm", type=int)
    ap.add_argument("--Lds", required=False, help="Lipschitz constant to be used in the minigolf task.", type=float)
    ap.add_argument("--file", required=False, help="True if you want to save data on a file", type=bool)
    args = vars(ap.parse_args())
    chooser(args)
