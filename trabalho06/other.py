import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def fourier_series(n_max, x):
    def bk(k):
        return 1 / ((2 * k - 1) ** 2)

    def wk(k):
        return (2 * k - 1) * np.pi

    resp = 0
    for n in range(1, n_max):
        resp += bk(n) * np.sin(wk(n) * x)
    return .5 - (4 / (np.pi ** 2)) * resp


def draw_fourier_series(armonics):
    x = np.linspace(-4, 4, 1000)
    f = np.array([fourier_series(armonics, i) for i in x])

    plt.plot(x, f, color="red", label="Série de Fourier")
    plt.title(f"Série de Fourier com {armonics} senos")
    plt.legend()
    plt.show()


draw_fourier_series(1)
draw_fourier_series(10)
draw_fourier_series(100)
draw_fourier_series(1000)
