from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(samples, intervals, title, minsp, maxsp):
    plt.figure()
    plt.title(title)
    [plt.plot(s[0], s[1], 'r.') for s in samples]

    # first dimension
    for i in intervals[0]:
        plt.plot((i[1], i[1]), (minsp[1], maxsp[1]))

    # second dimension
    for i in intervals[1]:
        plt.plot((minsp[0], maxsp[0]), (i[1], i[1]))

    plt.show()
    # plt.pause(1)


def plot_abstract_policy(intervals, abs_opt, theta, theta_1, theta_opt, fign):
    fig = plt.figure(fign + 1)
    plt.clf()
    ax = plt.axes(projection="3d")

    x_pos = [i[0] for i in intervals[0] for l in range(len(intervals[1]))]
    y_pos = [i[0] for l in range(len(intervals[0])) for i in intervals[1]]
    z_pos = [0] * (len(intervals[0]) * len(intervals[1]))
    x_size = [i[1] - i[0] - 0.001 for i in intervals[0] for l in range(len(intervals[1]))]
    y_size = [i[1] - i[0] - 0.001 for l in range(len(intervals[0])) for i in intervals[1]]
    for x in range(len(abs_opt)):
        if abs_opt[x] is None:
            abs_opt[x] = [0]
    z_size = [a[0] for a in abs_opt]

    plot_plane([-1, 1], [-2, 2], theta, ax, 'green')
    plot_plane([-1, 1], [-2, 2], theta_1, ax, 'blue')
    # plot_plane([-1, 1], [-1, 1], theta_opt, ax, 'red')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color='aqua')
    # supposed_to_be_best_policy(ax, intervals, abs_opt, theta, theta_1, theta_opt, fign)

    plt.draw()
    plt.pause(0.001)
    fig.show()


# def supposed_to_be_best_policy(ax, intervals, abs_opt, theta, theta_1, theta_opt, fign):
#     # ax = plt.axes(projection="3d")
#
#     x_pos = [i[0] for i in intervals[0] for l in range(len(intervals[1]))]
#     y_pos = [i[0] for l in range(len(intervals[0])) for i in intervals[1]]
#     z = [np.array(x).mean() * (-1.38) + np.array(y).mean() * (-0.82) for x in intervals[0] for y in intervals[1]]
#     z_pos = [0] * (len(intervals[0]) * len(intervals[1]))
#     x_size = [i[1] - i[0] - 0.001 for i in intervals[0] for l in range(len(intervals[1]))]
#     y_size = [i[1] - i[0] - 0.001 for l in range(len(intervals[0])) for i in intervals[1]]
#     z_size = z
#
#     ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color='red')


def plot_plane(x, y, param, ax, color):
    X, Y = np.meshgrid(x, y)
    Z = X * param[0][0] + Y * param[0][1]
    ax.plot_wireframe(X, Y, Z, color=color)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    color=color)


# --- lqg1d ---
def plot_lqg_abs_pol(intervals, abs_pol, theta, theta_1, theta_opt):
    fig = plt.figure()
    intervals = intervals[0]
    theta = theta[0][0]
    theta_1 = theta_1[0][0]
    theta_opt = theta_opt[0][0]
    for i, p in zip(intervals, abs_pol):
        plt.hlines(p, i[0], i[1])
    x = np.linspace(-2, 2, 3)
    plt.plot(x, theta * x, 'g')
    plt.plot(x, theta_1 * x, 'b')
    plt.plot(x, theta_opt * x, 'tab:orange')
    plt.show()

