import DPO.safetygym.safetygym_DPO as safe_main
import DPO.minigolf.minigolf_radialbasis as mini_main
# import DPO.mass.main_mass as mass_main
import sys


def main(task, seed, runs):
    if task == "mini":
        main = mini_main
    elif task == "mass":
        # main = mass_main
        pass
    elif task == "safe":
        main = safe_main
    for i in range(seed, seed+runs):
        main.main(i)


if __name__ == "__main__":
    try:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))  # task, seed, # iterations
    except:
        print("Please specify the task(mini, mass, safe), the seed and the number of runs.")
