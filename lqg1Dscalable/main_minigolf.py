import lqg1Dscalable.minigolf_radialbasis as mini_main
import numpy as np
from lqg1Dscalable.visualizer.lqg1d_visualizer import Lqg1dVisualizer

N_ITERATIONS = 6

stats = {}
avg = {}
std = {}
stats['param'] = np.array([])
stats['j'] = np.array([])
stats['sampleJ'] = np.array([])
stats['abstractJ'] = np.array([])

for i in range(1, N_ITERATIONS):
    data, optP, optJ = lqg_main.main(i)
    if i == 1:
        stats['param'] = np.array([data['param']])
        stats['j'] = np.array([data['j']])
        stats['sampleJ'] = np.array([data['sampleJ']])
        stats['abstractJ'] = np.array([data['abstractJ']])
    else:
        stats['param'] = np.append(stats['param'], [data['param']], axis=0)
        stats['j'] = np.append(stats['j'], [data['j']], axis=0)
        stats['sampleJ'] = np.append(stats['sampleJ'], [data['sampleJ']], axis=0)
        stats['abstractJ'] = np.append(stats['abstractJ'], [data['abstractJ']], axis=0)

avg['param'] = np.average(stats['param'], axis=0)
avg['j'] = np.average(stats['j'], axis=0)
avg['sampleJ'] = np.average(stats['sampleJ'], axis=0)
avg['abstractJ'] = np.average(stats['abstractJ'], axis=0)

std['param'] = np.std(stats['param'], axis=0)
std['j'] = np.std(stats['j'], axis=0)
std['sampleJ'] = np.std(stats['sampleJ'], axis=0)
std['abstractJ'] = np.std(stats['abstractJ'], axis=0)

visualizer = Lqg1dVisualizer("average values", "stats.jpg", opt_par=optP, optJ=optJ)
visualizer.clean_panels()
visualizer.show_average(avg, std)
visualizer.save_image()
