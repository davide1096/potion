import matplotlib.pyplot as plt
import os

class Visualizer(object):

    def __init__(self, title, filename, fig_num, size=None):
        super().__init__()
        self.title = title
        self.filename = filename
        self.fig_num = fig_num
        self.fig = plt.figure(self.fig_num, figsize=(12.8, 9.6) if size is None else size)

    def save_image(self):
        plt.figure(self.fig_num)
        filename = "../images/" + self.filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=150, transparent=False)
        plt.close(self.fig_num)

    def clean_panels(self):
        plt.figure(self.fig_num)
        plt.clf()
