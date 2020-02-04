import DPO.mass.main_mass_DPO as main_dpo
import DPO.mass.reinforce.REINOFRCE_mass as main_rei
import numpy as np
from DPO.visualizer.minigolf_visualizer import MGVisualizer
import csv
import os

ALGORITHM = "REINFORCE"

N_ITERATIONS = 3

ALPHA = [0.025]
LAM = [0.0001, 0.0005, 0.001] if ALGORITHM == "DPO" else [-4., -3., -2.]

tot_j_keeper = {}

for alpha in ALPHA:

    tot_j_keeper[alpha] = {}
    for lam in LAM:

        stats = {}
        avg = {}
        std = {}

        tot_env_j = 0
        tot_est_j = 0
        for i in range(1, N_ITERATIONS + 1):
            data, envj, estj = main_dpo.main(i, alpha, lam) if ALGORITHM == "DPO" else main_rei.main(i, alpha, lam)

            tot_env_j += envj
            tot_est_j += estj
            if i == 1:
                stats['param'] = np.array([data['param']])
                stats['j'] = np.array([data['j']])
                stats['estj'] = np.array([data['estj']])
            else:
                stats['param'] = np.append(stats['param'], [data['param']], axis=0)
                stats['j'] = np.append(stats['j'], [data['j']], axis=0)
                stats['estj'] = np.append(stats['estj'], [data['estj']], axis=0)

        for k in stats.keys():
            avg[k] = np.average(stats[k], axis=0)
            std[k] = np.std(stats[k], axis=0)

        filename_csv = "../csv/mass/DPO/ALPHA={}/LAM={}/stats.csv".format(alpha, lam) if ALGORITHM == "DPO" else \
            "../csv/mass/REINFORCE/ALPHA={}/LOGSIG={}/stats.csv".format(alpha, lam)
        os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
        data_file = open(filename_csv, mode='w')
        file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        file_writer.writerow(['avg param0', 'avg param1', 'avg j', 'avg est j', 'std param0', 'std param1', 'std j',
                              'std est j'])
        for i in range(len(avg['j'])):
            file_writer.writerow([avg['param'][i][0][0], avg['param'][i][0][1], avg['j'][i], avg['estj'][i],
                                  std['param'][i][0][0], std['param'][i][0][1], std['j'][i], std['estj'][i]])

        tot_j_keeper[alpha][lam] = [tot_env_j/N_ITERATIONS, tot_est_j/N_ITERATIONS]
        data_file.close()

filename = "../tuning/mass/DPO/alpha_lam_j.csv" if ALGORITHM == "DPO" else "../tuning/mass/REINFORCE/alpha_lam_j.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
data_file = open(filename, mode='w')
file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
file_writer.writerow(['alpha', 'lambda', 'j', 'estj']) if ALGORITHM == "DPO" else \
    file_writer.writerow(['alpha', 'logsig', 'j', 'estj'])
for alpha in ALPHA:
    for lam in LAM:
        file_writer.writerow([alpha, lam, tot_j_keeper[alpha][lam][0], tot_j_keeper[alpha][lam][1]])
data_file.close()
