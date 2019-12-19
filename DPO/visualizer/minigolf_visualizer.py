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
        self.title = title
        self.filename = filename
        plt.figure(figsize=(12.8, 9.6))

    def show_values(self, w, j):

        plt.clf()
        self.w1.append(w[0])
        self.w2.append(w[1])
        self.w3.append(w[2])
        self.w4.append(w[3])
        self.j.append(j)
        plt.suptitle(self.title)

        plt.subplots_adjust(hspace=0.6, wspace=0.4)

        plt.subplot(3, 2, 1)
        plt.title("W1")
        plt.plot(self.w1, label="W1")
        plt.annotate(np.round(self.w1[-1], decimals=3), (len(self.w1) - 1, self.w1[-1]))
        plt.grid(b=True)

        plt.subplot(3, 2, 2)
        plt.title("W2")
        plt.plot(self.w2, label="W2")
        plt.annotate(np.round(self.w2[-1], decimals=3), (len(self.w2) - 1, self.w2[-1]))
        plt.grid(b=True)

        plt.subplot(3, 2, 3)
        plt.title("W3")
        plt.plot(self.w3, label="W3")
        plt.annotate(np.round(self.w3[-1], decimals=3), (len(self.w3) - 1, self.w3[-1]))
        plt.grid(b=True)

        plt.subplot(3, 2, 4)
        plt.title("W4")
        plt.plot(self.w4, label="W4")
        plt.annotate(np.round(self.w4[-1], decimals=3), (len(self.w4) - 1, self.w4[-1]))
        plt.grid(b=True)

        plt.subplot(3, 2, 6)
        plt.title("J")
        plt.plot(self.j, label="J")
        plt.annotate(np.round(self.j[-1], decimals=3), (len(self.j) - 1, self.j[-1]))
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
        plt.suptitle("Averages and confidence")
        plt.subplots_adjust(hspace=0.6, wspace=0.4)

        x = [n * 10 for n in range(0, len(avg['w1']))]

        plt.subplot(3, 2, 1)
        plt.title("W1")
        plt.annotate(np.round(avg['w1'][-1], decimals=3), ((len(avg['w1']) - 1) * 10, avg['w1'][-1]))
        std_resized = np.resize([2*s for s in std['w1']], (len(std['w1']), ))
        plt.errorbar(x, avg['w1'], yerr=std_resized)
        plt.grid(b=True)

        plt.subplot(3, 2, 2)
        plt.title("W2")
        plt.annotate(np.round(avg['w2'][-1], decimals=3), ((len(avg['w2']) - 1) * 10, avg['w2'][-1]))
        std_resized = np.resize([2 * s for s in std['w2']], (len(std['w2']),))
        plt.errorbar(x, avg['w2'], yerr=std_resized)
        plt.grid(b=True)

        plt.subplot(3, 2, 3)
        plt.title("W3")
        plt.annotate(np.round(avg['w3'][-1], decimals=3), ((len(avg['w3']) - 1) * 10, avg['w3'][-1]))
        std_resized = np.resize([2 * s for s in std['w3']], (len(std['w3']),))
        plt.errorbar(x, avg['w3'], yerr=std_resized)
        plt.grid(b=True)

        plt.subplot(3, 2, 4)
        plt.title("W4")
        plt.annotate(np.round(avg['w4'][-1], decimals=3), ((len(avg['w4']) - 1) * 10, avg['w4'][-1]))
        std_resized = np.resize([2 * s for s in std['w4']], (len(std['w4']),))
        plt.errorbar(x, avg['w4'], yerr=std_resized)
        plt.grid(b=True)

        plt.subplot(3, 2, 6)
        plt.title("J")
        plt.annotate(np.round(avg['j'][-1], decimals=3), ((len(avg['j']) - 1) * 10, avg['j'][-1]))
        std_resized = np.resize([2 * s for s in std['j']], (len(std['j']),))
        plt.errorbar(x, avg['j'], yerr=std_resized)
        plt.grid(b=True)

        plt.draw()
        plt.pause(0.001)

