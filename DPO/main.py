import DPO.safetygym.safetygym_DPO as safe_main
import DPO.minigolf.minigolf_radialbasis as mini_main
import DPO.mass.mass_DPO as mass_main
import sys
import argparse


def chooser(args):
    task = args['task']
    seed = args['seed']
    runs = args['nruns']

    if task == "mini":
        main = mini_main
    elif task == "mass":
        main = mass_main
    elif task == "safe":
        main = safe_main
    for i in range(seed, seed+runs):
        main.main(i, args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Name of the task: mini, mass or safe", type=str)
    ap.add_argument("--seed", required=False, help="Value of the initial seed", default=1, type=int)
    ap.add_argument("--nruns", required=False, help="N. of runs of the task", default=10, type=int)
    ap.add_argument("--alpha", required=False, help="Learning rate used in policy projection", type=float)
    ap.add_argument("--lambda", required=False, help="Regularization used in policy projection", type=float)
    ap.add_argument("--mcrst", required=False, help="Number of abstract states on each state dimension", type=int)
    args = vars(ap.parse_args())
    chooser(args)
