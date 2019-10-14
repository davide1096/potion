import matplotlib.pyplot as plt

parameter = []


def initialization(a, b, det_par_opt, noise, init_par):
    title = "A = {}, B = {}, Optimal par = {}, Variance of noise = {}".format(a, b, det_par_opt, noise)
    plt.title(title)
    plt.xlabel("number of iterations")
    plt.ylabel("deterministic parameter")
    plt.ion()
    plt.show()
    parameter.append(init_par)


def show_new_value(par, par_opt):
    parameter.append(par)
    plt.plot(parameter)
    plt.hlines(par_opt, 0, len(parameter) - 1, colors='r', linestyles='dashed')
    plt.draw()
    plt.pause(0.001)


def save_img(tf_known, a, b, noise):
    key = "{}_{}_{}_{}_{}".format(a, b, noise, 1 if tf_known else 0, parameter[0])
    key = key.replace('.', ',')
    filename = "./images/img" + key + ".jpg"
    plt.savefig(filename, dpi=150, transparent=False)
