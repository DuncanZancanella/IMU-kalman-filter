import numpy as np

def inclinacao(t):
	return np.piecewise(t,
		[t < 10, (t >= 10) & (t < 40), t >= 40],
		[0, lambda t: (np.pi / 2) / (1 + np.exp(-(t-20))), lambda t: (np.pi / 2) * (1 - 1 / (1 + np.exp(-(t-50)))),],
	)


def gyro(t, sigma):
	d_inclincacao = np.gradient(inclincacao(t), t)
	ruido = np.random.normal(0, sigma, d_inclincacao.shape)
	return d_inclincacao + ruido


def acc(t, sigma):
	ruido = np.random.normal(0, sigma, t.shape())
	return inclinacao(t) + ruido

gyro = 3
acc = 4



import matplotlib.pyplot as plt


def plot_function(func, xmin=0, xmax=80, num=1000, ax=None, **plot_kwargs):
	x = np.linspace(xmin, xmax, num)
	y = np.degrees(func(x))

	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure

	ax.plot(x, y, **plot_kwargs)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.grid(True)

	plt.show()
	return



plot_function(inclinacao)