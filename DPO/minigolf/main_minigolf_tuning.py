import DPO.minigolf.DPO.minigolf_radialbasis as mini_main_dpo
import DPO.minigolf.REINFORCE.REINFORCE2 as mini_main_rei
import numpy as np
from DPO.visualizer.minigolf_visualizer import MGVisualizer
import csv
import os

algorithm = "DPO"  # use "DPO" or "REINFORCE" to change algorithm
N_ITERATIONS = 5

ALPHA = [0.005, 0.01, 0.05]
LAM = [0.0005, 0.001, 0.005]

tot_j_keeper = {}

for alpha in ALPHA:

    tot_j_keeper[alpha] = {}
    for lam in LAM:

        stats = {}
        avg = {}
        std = {}

        tot_j = 0
        for i in range(1, N_ITERATIONS + 1):
            data, j = mini_main_dpo.main(i, alpha, lam) if algorithm == "DPO" else mini_main_rei.main(i)
            tot_j += j
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

        filename = "minigolf/DPO/ALPHA={}/LAM={}/stats.png".format(alpha, lam) if algorithm == "DPO" \
            else "/minigolf/REINFORCE/stats.png"
        filename_csv = "../csv/minigolf/DPO/ALPHA={}/LAM={}/stats.csv".format(alpha, lam)
        os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
        data_file = open(filename_csv, mode='w')
        file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        file_writer.writerow(['avg w1', 'avg w2', 'avg w3', 'avg w4', 'avg j', 'avg fail', 'std w1', 'std w2', 'std w3',
                             'std w4', 'std j', 'std fail'])
        for i in range(len(avg['w1'])):
            file_writer.writerow([avg['w1'][i], avg['w2'][i], avg['w3'][i], avg['w4'][i], avg['j'][i], avg['fail'][i],
                                  std['w1'][i], std['w2'][i], std['w3'][i], std['w4'][i], std['j'][i], std['fail'][i]])

        visualizer = MGVisualizer("average values", filename)
        visualizer.clean_panels()
        visualizer.show_average(avg, std)
        visualizer.save_image()

        tot_j_keeper[alpha][lam] = tot_j / N_ITERATIONS

filename = "../tuning/minigolf/DPO/alpha_lam_j.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
data_file = open(filename, mode='w')
file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
file_writer.writerow(['alpha', 'lambda', 'j'])
for alpha in ALPHA:
    for lam in LAM:
        file_writer.writerow([alpha, lam, tot_j_keeper[alpha][lam]])

