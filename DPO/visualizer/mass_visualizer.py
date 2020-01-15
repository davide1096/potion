import matplotlib.pyplot as plt
from DPO.visualizer.visualizer import Visualizer
import numpy as np
from collections import OrderedDict


class MassVisualizer(Visualizer):

    def __init__(self, title, filename, init_par=None, opt_par=None, initJ=None, optJ=None):

        super().__init__()
        self.par = []
        self.J = []
        self.estJ = []
        self.title = title
        self.opt_par = opt_par
        self.optJ = optJ
        self.filename = filename
        if init_par is not None:
            self.par.append(init_par)
        if initJ is not None:
            self.J.append(initJ)

    def show_values(self, new_par, newJ, new_estJ):

        # f = plt.figure(0)
        plt.clf()
        self.par.append(new_par)
        self.J.append(newJ)
        self.estJ.append(new_estJ)
        plt.suptitle(self.title)

        for i in range(new_par.size):
            plt.subplot(2, 2, i+1)
            plt.title("Deterministic parameter")
            plt.plot([p[0][i] for p in self.par])
            plt.annotate(np.round(self.par[-1][0][i], decimals=3), (len(self.par) - 1, self.par[-1][0][i]))
            if self.opt_par is not None:
                plt.hlines(self.opt_par[0][i], 0, len(self.par) - 1, colors='r', linestyles='dashed')

        plt.subplots_adjust(hspace=0.5)

        plt.subplot(2, 2, 3)
        plt.title("Performance measures")
        plt.plot(self.J, 'b', label="J")
        plt.annotate(np.round(self.J[-1], decimals=3), (len(self.J) - 1, self.J[-1]))
        if len(self.estJ) > 1:
            plt.plot(range(1, len(self.estJ) + 1), self.estJ, 'g', label="J from samples - det policy")
            plt.annotate(np.round(self.estJ[-1], decimals=3), (len(self.estJ), self.estJ[-1]))
        if self.optJ is not None:
            plt.hlines(self.optJ, 0, len(self.J) - 1, colors='r', linestyles='dashed')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.draw()
        plt.pause(0.001)
        # f.show()

    def save_image(self):
        filename = "../images/mass/" + self.filename
        plt.savefig(filename, dpi=150, transparent=False)
        plt.close()

    def clean_panels(self):
        plt.clf()
