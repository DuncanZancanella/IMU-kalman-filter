import numpy as np

class Kalman:
	"""
	Implementa o filtro de Kalman para sistemas lineares discretos.

	O algoritmo estima recursivamente o estado de um sistema dinâmico
	linear a partir de um modelo de transição de estados e de medições
	sujeitas a ruído.

	O modelo considerado é

	.. math::

		x_k = F x_{k-1} + B u_k + w_k

		z_k = H x_k + v_k

	onde :math:`w_k \\sim \\mathcal{N}(0, R)` representa o ruído do
	processo e :math:`v_k \\sim \\mathcal{N}(0, Q)` representa o ruído
	da observação.

	Parameters
	----------
	F : ndarray da forma (n, n)
		Matriz de transição de estados.

	B : ndarray da forma (n, m)
		Matriz de controle.

	H : ndarray da forma (p, n)
		Matriz de observação.

	Q : ndarray da forma (p, p)
		Matriz de covariância do ruído da observação.

	R : ndarray da forma (n, n)
		Matriz de covariância do ruído do processo.

	P0 : ndarray da forma (n, n)
		Covariância inicial da estimativa do estado.

	x0 : ndarray da forma (n, 1)
		Estimativa inicial do estado.

	Attributes
	----------
	mu : ndarray
		Estimativa corrente do estado.

	S : ndarray
		Covariância corrente da estimativa.

	K : ndarray
		Ganho de Kalman calculado durante a etapa de correção.
	"""
	def __init__(self, F, B, H, Q, R, P0, x0):
		self.F = F
		self.B = B
		self.H = H
		self.Q = Q
		self.R = R
		self.P0 = P0
		self.x0 = x0
		self.mu = x0
		self.S = P0
		self.K = 0.5

	def predict(self, u):
		"""
		Executa a etapa de predição do filtro de Kalman.

		Atualiza a estimativa do estado e sua covariância utilizando
		apenas o modelo do sistema e o sinal de controle.

		Parameters
		----------
		u : ndarray da forma (m, 1)
			Vetor de entrada (controle) aplicado ao sistema.

		Returns
		-------
		mu : ndarray da forma (n, 1)
			Estado predito.

		S : ndarray da forma (n, n)
			Covariância da estimativa predita.
		"""
		self.mu = self.F @ self.mu + self.B @ u
		self.S = self.F @ self.S @ self.F.T + self.R
		return self.mu, self.S

	def update(self, z):
		"""
		Executa a etapa de correção do filtro de Kalman.

		Atualiza a estimativa do estado utilizando uma nova medição.

		Parameters
		----------
		z : ndarray da forma (p, 1)
			Vetor de observações.

		Returns
		-------
		mu : ndarray da forma (n, 1)
			Estado corrigido.

		S : ndarray da forma (n, n)
			Covariância corrigida da estimativa.
		"""
		self.K = self.S @ self.H.T @ np.linalg.inv( self.H @ self.S @ self.H.T + self.Q)
		self.mu = self.mu + self.K @ ( z - self.H @ self.mu )
		self.S = self.S - self.K @ self.H @ self.S
		return self.mu, self.S