import matplotlib.pyplot as plt
from DPO.visualizer.visualizer import Visualizer
import numpy as np
from collections import OrderedDict


class Lqg1dVisualizer(Visualizer):

    def __init__(self, title, filename, init_par=None, opt_par=None, initJ=None, optJ=None):

        super().__init__()
        self.par = []
        self.J = []
        self.estJ = []
        self.estAbsJ = []
        self.title = title
        self.opt_par = opt_par
        self.optJ = optJ
        self.filename = filename
        if init_par is not None:
            self.par.append(init_par)
        if initJ is not None:
            self.J.append(initJ)

    def show_values(self, new_par, newJ, new_estJ, new_est_absJ):

        plt.clf()
        self.par.append(new_par)
        self.J.append(newJ)
        self.estJ.append(new_estJ)
        self.estAbsJ.append(new_est_absJ)
        plt.suptitle(self.title)

        plt.subplot(2, 1, 1)
        plt.title("Deterministic parameter")
        plt.plot(self.par)
        plt.annotate(np.round(self.par[-1], decimals=3), (len(self.par) - 1, self.par[-1]))
        if self.opt_par is not None:
            plt.hlines(self.opt_par, 0, len(self.par) - 1, colors='r', linestyles='dashed')

        plt.subplots_adjust(hspace=0.5)

        plt.subplot(2, 1, 2)
        plt.title("Performance measures")
        plt.plot(self.J, 'b', label="J")
        plt.annotate(np.round(self.J[-1], decimals=3), (len(self.J) - 1, self.J[-1]))
        if len(self.estAbsJ) > 1:
            plt.plot(range(1, len(self.estJ) + 1), self.estJ, 'g', label="J from samples - det policy")
            plt.annotate(np.round(self.estJ[-1], decimals=3), (len(self.estJ), self.estJ[-1]))
            plt.plot(range(1, len(self.estAbsJ) + 1), self.estAbsJ, 'tab:orange', label="J from samples - abs policy")
            plt.annotate(np.round(self.estAbsJ[-1], decimals=3), (len(self.estAbsJ), self.estAbsJ[-1]))
        if self.optJ is not None:
            plt.hlines(self.optJ, 0, len(self.J) - 1, colors='r', linestyles='dashed')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.draw()
        plt.pause(0.001)

    def save_image(self):
        filename = "../images/" + self.filename
        plt.savefig(filename, dpi=150, transparent=False)
        plt.close()

    def clean_panels(self):
        plt.clf()

    def show_average(self, avg, std):

        plt.figure(figsize=(12.8, 9.6))
        plt.suptitle("Averages and confidence")
        plt.subplots_adjust(hspace=0.4, wspace=0.2)

        plt.subplot(2, 2, 1)
        plt.title("Deterministic parameter")
        plt.annotate(np.round(avg['param'][-1], decimals=3), (len(avg['param']) - 1, avg['param'][-1]))
        std_resized = np.resize([2*s for s in std['param']], (len(std['param']), ))
        plt.errorbar(range(0, len(avg['param'])), avg['param'], yerr=std_resized)
        if self.opt_par is not None:
            plt.hlines(self.opt_par, 0, len(avg['param']) - 1, colors='r', linestyles='dashed')

        plt.subplot(2, 2, 3)
        plt.title("Performance measures")
        plt.annotate(np.round(avg['j'][-1], decimals=3), (len(avg['j']) - 1, avg['j'][-1]))
        plt.plot(avg['j'], 'b', label="J")
        plt.plot(range(1, len(avg['sampleJ']) + 1), avg['sampleJ'], 'g', label="J from samples - det policy")
        plt.plot(range(1, len(avg['abstractJ']) + 1), avg['abstractJ'], 'tab:orange',
                 label="J from samples - abs policy")
        if self.optJ is not None:
            plt.hlines(self.optJ, 0, len(avg['j']) - 1, colors='r', linestyles='dashed')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.subplot(2, 2, 2)
        plt.title("Performance from samples - det policy")
        plt.annotate(np.round(avg['sampleJ'][-1], decimals=3), (len(avg['sampleJ']), avg['sampleJ'][-1]))
        std_resized = np.resize([2*s for s in std['sampleJ']], (len(std['sampleJ']),))
        plt.errorbar(range(1, len(avg['sampleJ']) + 1), avg['sampleJ'], yerr=std_resized, color='green')

        plt.subplot(2, 2, 4)
        plt.title("Performance from samples - abs policy")
        plt.annotate(np.round(avg['abstractJ'][-1], decimals=3), (len(avg['abstractJ']), avg['abstractJ'][-1]))
        std_resized = np.resize([2*s for s in std['abstractJ']], (len(std['abstractJ']),))
        plt.errorbar(range(1, len(avg['abstractJ']) + 1), avg['abstractJ'], yerr=std_resized, color='orange')

        plt.draw()
        plt.pause(0.001)

