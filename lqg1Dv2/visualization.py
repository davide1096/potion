import matplotlib.pyplot as plt

parameter = []


def initialization(a, b, det_par_opt, noise):
    title = "A = {}, B = {}, Optimal par = {}, Variance of noise = {}".format(a, b, det_par_opt, noise)
    plt.title(title)
    plt.xlabel("number of iterations")
    plt.ylabel("deterministic parameter")
    plt.ion()
    plt.show()


def show_new_value(par):
    parameter.append(par)
    plt.plot(parameter)
    plt.draw()
    plt.pause(0.001)


def save_img(tf_known, a, b, noise):
    key = "{}{}{}{}".format(a, b, noise, 1 if tf_known else 0)
    filename = "./images/img" + key + ".jpg"
    plt.savefig(filename, dpi=150, transparent=False)
