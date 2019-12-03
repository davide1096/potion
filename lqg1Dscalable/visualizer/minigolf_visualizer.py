import matplotlib.pyplot as plt
from lqg1Dscalable.visualizer.visualizer import Visualizer
import numpy as np
from collections import OrderedDict


class MGVisualizer(Visualizer):

    def __init__(self, title, filename):

        super().__init__()
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.w4 = []
        self.w5 = []
        self.b = []
        self.j = []
        self.title = title
        self.filename = filename

    def show_values(self, w, b, j):

        self.w1.append(w[0])
        self.w2.append(w[1])
        self.w3.append(w[2])
        self.w4.append(w[3])
        self.w5.append(w[4])
        self.b.append(b)
        self.j.append(j)
        plt.suptitle(self.title)

        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        plt.subplot(4, 2, 1)
        plt.title("W1")
        plt.plot(self.w1, label="W1")

        plt.subplot(4, 2, 2)
        plt.title("W2")
        plt.plot(self.w2, label="W2")

        plt.subplot(4, 2, 3)
        plt.title("W3")
        plt.plot(self.w3, label="W3")

        plt.subplot(4, 2, 4)
        plt.title("W4")
        plt.plot(self.w4, label="W4")

        plt.subplot(4, 2, 5)
        plt.title("W5")
        plt.plot(self.w5, label="W5")

        plt.subplot(4, 2, 6)
        plt.title("b")
        plt.plot(self.b, label="b")

        plt.subplot(4, 2, 7)
        plt.title("J")
        plt.plot(self.j, label="J")

        plt.draw()
        plt.pause(0.001)

    def save_image(self):
        filename = "./images/" + self.filename
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
        std_resized = np.resize([2*s for s in std['param']], (len(std['param']), ))
        plt.errorbar(range(0, len(avg['param'])), avg['param'], yerr=std_resized)
        if self.opt_par is not None:
            plt.hlines(self.opt_par, 0, len(avg['param']) - 1, colors='r', linestyles='dashed')

        plt.subplot(2, 2, 3)
        plt.title("Performance measures")
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
        std_resized = np.resize([2*s for s in std['sampleJ']], (len(std['sampleJ']),))
        plt.errorbar(range(1, len(avg['sampleJ']) + 1), avg['sampleJ'], yerr=std_resized, color='green')

        plt.subplot(2, 2, 4)
        plt.title("Performance from samples - abs policy")
        std_resized = np.resize([2*s for s in std['abstractJ']], (len(std['abstractJ']),))
        plt.errorbar(range(1, len(avg['abstractJ']) + 1), avg['abstractJ'], yerr=std_resized, color='orange')

        plt.draw()
        plt.pause(0.001)

