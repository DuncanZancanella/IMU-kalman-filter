import numpy as np
from kalman import Kalman

def inclinacao(t):
	return np.piecewise(t,
		[t < 10, (t >= 10) & (t < 40), t >= 40],
		[0, lambda t: (np.pi / 2) / (1 + np.exp(-(t-20))), lambda t: (np.pi / 2) * (1 - 1 / (1 + np.exp(-(t-50)))),],
	)


def gyro(t, sigma):
	d_inclincacao = np.gradient(inclinacao(t), t)
	ruido = np.random.normal(1, sigma, d_inclincacao.shape)
	return d_inclincacao + ruido


def acc(t, sigma):
	ruido = np.random.normal(0, sigma, t.shape)
	return inclinacao(t) + ruido


dt = 0.01 # passo de tempo

F = np.array([
	[1, dt],
	[0, 1]
])

B = np.zeros((2,1)) # sem controle

H = np.array([[1,1]]) # apenas ângulo

sigma_acc = 0.01 # ruído da medição do acelerômetro

Q = np.array([[sigma_acc**2]])

sigma_gyro = 0.01 # ruído do modelo

R = np.array([
	[0,0],
	[0,sigma_gyro**2]
])

P0 = np.eye(2)

x0 = np.array([[0],[0]])


kalman = Kalman(F, B, H, Q, R, P0, x0)

T = 60

t = np.arange(0, T, dt)

theta_real = inclinacao(t)

gyro_measure = gyro(t, 0.01)

acc_measure = acc(t, 0.05)

estimativas = []


for k in range(len(t)):

	# previsão usando o giroscópio
	omega = gyro_measure[k]

	u = np.array([
		[omega]
	])

	mu_pred, S_pred = kalman.predict(u)


	# correção usando acelerômetro
	z = np.array([
		[acc_measure[k]]
	])

	mu, S = kalman.update(z)

	estimativas.append(mu.flatten())


import matplotlib.pyplot as plt


estimativas = np.array(estimativas)


plt.figure(figsize=(10,4))

plt.plot(
	t,
	theta_real,
	label="Real"
)

plt.plot(
	t,
	acc_measure,
	alpha=0.5,
	label="Acelerômetro"
)

plt.plot(
	t,
	estimativas[:,0],
	label="Kalman"
)


plt.xlabel("Tempo (s)")
plt.ylabel("Ângulo (rad)")
plt.grid()
plt.legend()

plt.show()




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