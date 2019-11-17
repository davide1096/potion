import matplotlib.pyplot as plt


def plot_bounds(bounds1, title):
    seg = [[(b[0], b[1]), (i, i)] for i, b in enumerate(bounds1)]
    plt.title(title)
    [plt.plot(s[0], s[1]) for s in seg]

    plt.show()
    # plt.pause(1)


# bounds = [[1, 2], [1.5, 3], [2, 4]]
# plot_bounds(bounds)
