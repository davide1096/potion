import matplotlib.pyplot as plt
from DPO.visualizer.visualizer import Visualizer
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import os


class MGVisualizer(Visualizer):

    def __init__(self, title, filename, fig_num):

        super().__init__(title, filename, fig_num)
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.w4 = []
        self.j = []
        self.fail = []
        # self.fig = plt.figure(self.fig_num, figsize=(12.8, 9.6))
        self.k = ['w1', 'w2', 'w3', 'w4', 'fail', 'j']
        self.titles = ["w1", "w2", "w3", "w4", "Failures", "Performance"]

    def show_values(self, w, j, fail):
        plt.figure(self.fig_num)
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
        self.fig.canvas.start_event_loop(0.001)

    # def save_image(self):
    #     filename = "../images/" + self.filename
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     plt.savefig(filename, dpi=150, transparent=False)
    #     plt.close()

    def show_average(self, avg, std):

        fig = plt.figure(figsize=(12.8, 9.6))
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
        self.fig.canvas.start_event_loop(0.001)
        plt.close()

def plt_regr_perf(w, e):
    plt.figure(2)
    plt.clf()
    fig = plt.figure()
    for i in range(12):
        w_cur = w[i]
        e_cur = e[i]
        for k in range(3):
            ax = fig.add_subplot(12, 4, i*4 + k + 1)
            wd = [w_cur[i][k] for i in range(len(w_cur))]
            ax.plot(wd)
        ax = fig.add_subplot(12, 4, (i+1)*4)
        ax.plot(e_cur)


class RegrVisualizer(Visualizer):

    def __init__(self, title, filename, fig_num, size=None):
        super().__init__(title, filename, fig_num, size)
        self.w = [[[] for i in range(3)] for i in range(12)]
        self.e = [[] for i in range(12)]

    def show_values(self, w, e):
        fig = plt.figure(self.fig_num)
        fig.clf()

        for i in range(12):
            for k in range(3):
                self.w[i][k].append(w[i][k])
            self.e[i].append(e[i])

        fig.suptitle(self.title)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        for i in range(12):
            for k in range(3):
                ax = plt.subplot(12, 4, i * 4 + k + 1)
                ax.plot(self.w[i][k])
                ax.grid(b=True)
                if i==0:
                    if k==0:
                        ax.title.set_text("Intercept")
                    elif k==1:
                        ax.title.set_text("W_S")
                    else:
                        ax.title.set_text("W_A")
                if k==0:
                    pad = 5
                    ax.annotate("MCRST {}".format(i), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                                xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right',
                                va='center')
            ax = plt.subplot(12, 4, (i + 1) * 4)
            ax.plot(self.e[i])
            plt.text(1.02, 0.9, np.round(self.e[i][-1], decimals=3), transform=ax.transAxes)
            ax.grid(b=True)
            if i==0:
                ax.title.set_text("Error")
        plt.tight_layout()
        plt.draw()
        fig.canvas.start_event_loop(0.001)


class ThreeDVisualizer(Visualizer):
    def __init__(self, title, filename, fig_num, size=None):
        super().__init__(title, filename, fig_num, size)

    def show_values(self, w, it, maxit, int):
        fig = plt.figure(self.fig_num)

        if it%50==0:
            # s = np.linspace(0, 20, 10)
            a = np.linspace(0, 5, 10)
            for i in range(12):
                s = np.linspace(int[i+1][0], int[i+1][1], 10)
                S, A = np.meshgrid(s, a)
                ax = plt.subplot(12, (maxit//50)+1, i*(maxit//50 + 1) + it//50 + 1, projection='3d')
                Z = w[i][0] + w[i][1]*S + w[i][2]*A
                ax.plot_wireframe(S, A, Z)
                ax.set_zlim([-10,+10])
                if i==0:
                    ax.title.set_text("Iter {}".format(it))
            plt.tight_layout()



