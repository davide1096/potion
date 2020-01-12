import matplotlib.pyplot as plt
from DPO.visualizer.visualizer import Visualizer
import numpy as np
from collections import OrderedDict


class MGVisualizer(Visualizer):

    def __init__(self, title, filename):

        super().__init__()
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.w4 = []
        self.j = []
        self.fail = []
        self.title = title
        self.filename = filename
        plt.figure(figsize=(12.8, 9.6))
        self.k = ['w1', 'w2', 'w3', 'w4', 'fail', 'j']
        self.titles = ["w1", "w2", "w3", "w4", "Failures", "Performance"]

    def show_values(self, w, j, fail):

        plt.clf()
        self.w1.append(w[0])
        self.w2.append(w[1])
        self.w3.append(w[2])
        self.w4.append(w[3])
        self.j.append(j)
        self.fail.append(fail)
        plt.suptitle(self.title)

        plt.subplots_adjust(hspace=0.6, wspace=0.4)

        var = [self.w1, self.w2, self.w3, self.w4, self.fail, self.j]
        for i in range(6):
            ax = plt.subplot(3, 2, i+1)
            plt.title(self.titles[i])
            plt.plot(var[i], label=self.titles[i])
            plt.text(1.02, 0.9, np.round(var[i][-1], decimals=3), transform=ax.transAxes)
            plt.grid(b=True)

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
        plt.suptitle("Average and confidence")
        plt.subplots_adjust(hspace=0.6, wspace=0.4)

        x = [n * 10 for n in range(0, len(avg['w1']))]

        for i in range(6):
            ax = plt.subplot(3, 2, i+1)
            plt.title(self.titles[i])
            plt.text(1.02, 0.9, np.round(avg[self.k[i]][-1], decimals=3), transform=ax.transAxes)
            std_resized = np.resize([2*s for s in std[self.k[i]]], (len(std[self.k[i]]), ))
            plt.errorbar(x, avg[self.k[i]], yerr=std_resized)
            plt.grid(b=True)

        plt.draw()
        plt.pause(0.001)

