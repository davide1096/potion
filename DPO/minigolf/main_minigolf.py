import DPO.minigolf.DPO.minigolf_radialbasis as mini_main_dpo
import DPO.minigolf.REINFORCE.REINFORCE2 as mini_main_rei
import numpy as np
from DPO.visualizer.minigolf_visualizer import MGVisualizer

algorithm = "DPO"  # use "DPO" or "REINFORCE" to change algorithm
N_ITERATIONS = 5

stats = {}
avg = {}
std = {}

for i in range(1, N_ITERATIONS + 1):
    data = mini_main_dpo.main(i) if algorithm == "DPO" else mini_main_rei.main(i)
    if i == 1:
        stats['w1'] = np.array([data['w1']])
        stats['w2'] = np.array([data['w2']])
        stats['w3'] = np.array([data['w3']])
        stats['w4'] = np.array([data['w4']])
        stats['j'] = np.array([data['j']])
        stats['fail'] = np.array([data['fail']])
    else:
        stats['w1'] = np.append(stats['w1'], [data['w1']], axis=0)
        stats['w2'] = np.append(stats['w2'], [data['w2']], axis=0)
        stats['w3'] = np.append(stats['w3'], [data['w3']], axis=0)
        stats['w4'] = np.append(stats['w4'], [data['w4']], axis=0)
        stats['j'] = np.append(stats['j'], [data['j']], axis=0)
        stats['fail'] = np.append(stats['fail'], [data['fail']], axis=0)

avg['w1'] = np.average(stats['w1'], axis=0)
avg['w2'] = np.average(stats['w2'], axis=0)
avg['w3'] = np.average(stats['w3'], axis=0)
avg['w4'] = np.average(stats['w4'], axis=0)
avg['j'] = np.average(stats['j'], axis=0)
avg['fail'] = np.average(stats['fail'], axis=0)

std['w1'] = np.std(stats['w1'], axis=0)
std['w2'] = np.std(stats['w2'], axis=0)
std['w3'] = np.std(stats['w3'], axis=0)
std['w4'] = np.std(stats['w4'], axis=0)
std['j'] = np.std(stats['j'], axis=0)
std['fail'] = np.std(stats['fail'], axis=0)

filename = "/minigolf/DPO/stats.jpg" if algorithm == "DPO" else "/minigolf/REINFORCE/stats.jpg"
visualizer = MGVisualizer("average values", filename)
visualizer.clean_panels()
visualizer.show_average(avg, std)
visualizer.save_image()
