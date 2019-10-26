import matplotlib.pyplot as plt
from lqg1Dscalable.visualizer.visualizer import Visualizer


class Lqg1dVisualizer(Visualizer):

    def __init__(self, title, xlabel, ylab1, ylab2, filename, init_par=None, opt_par=None, initJ=None, optJ=None):

        super().__init__()
        self.par = []
        self.J = []
        self.estJ = []
        self.opt_par = opt_par
        self.optJ = optJ
        self.filename = filename
        if init_par is not None:
            self.par.append(init_par)
        if initJ is not None:
            self.J.append(initJ)

        plt.subplot(2, 1, 1)
        plt.title(title)
        plt.ylabel(ylab1)

        plt.subplot(2, 1, 2)
        plt.xlabel(xlabel)
        plt.ylabel(ylab2)

        plt.ion()
        plt.show()

    def show_values(self, new_par, newJ, new_estJ):

        self.par.append(new_par)
        self.J.append(newJ)
        self.estJ.append(new_estJ)

        plt.subplot(2, 1, 1)
        plt.plot(self.par)
        if self.opt_par is not None:
            plt.hlines(self.opt_par, 0, len(self.par) - 1, colors='r', linestyles='dashed')

        plt.subplot(2, 1, 2)
        plt.plot(self.J, 'b')
        if len(self.estJ) > 1:
            plt.plot(range(1, len(self.estJ) + 1), self.estJ, 'g')
        if self.optJ is not None:
            plt.hlines(self.optJ, 0, len(self.J) - 1, colors='r', linestyles='dashed')

        plt.draw()
        plt.pause(0.001)

    def save_image(self):
        filename = "./images/" + self.filename
        plt.savefig(filename, dpi=150, transparent=False)
