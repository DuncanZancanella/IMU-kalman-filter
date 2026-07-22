import serial
import numpy as np
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from kalman import Kalman


PORT = "COM5"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0.01)

# --- parâmetros filtro de Kalman --- #

dt = 0.01

F = np.array([
	[1, dt],
	[0, 1]
])

# gyro is the control input
B = np.array([ [dt], [0] ])

# accelerometer measures only angle
H = np.array([ [1, 0] ])

sigma_acc = 0.03
sigma_gyro = 0.01

Q = np.array([ [sigma_acc**2] ])

R = np.array([
	[0, 0],
	[0, sigma_gyro**2]
])

P0 = np.eye(2)

x0 = np.array([ [0.0], [0.0] ])

kalman = Kalman(F, B, H, Q,	R, P0, x0)

# --- --------------------------- --- #

N = 500

time_data = deque(maxlen=N)
gyro_data = deque(maxlen=N)
acc_data = deque(maxlen=N)
kalman_data = deque(maxlen=N)

t = 0

app = QtWidgets.QApplication([])

pg.setConfigOptions(antialias=True)

win = pg.GraphicsLayoutWidget(
	title="STM32 IMU + Kalman Filter"
)

plot = win.addPlot()

plot.addLegend()

plot.showGrid(x=True, y=True)

plot.setLabel("left", "Angle (rad)")
plot.setLabel("bottom", "Time (s)")

gyro_curve = plot.plot(
	pen='r',
	name="Gyroscope"
)

acc_curve = plot.plot(
	pen='g',
	name="Accelerometer"
)

kalman_curve = plot.plot(
	pen=pg.mkPen('b', width=3),
	name="Kalman"
)

win.show()

###########################################################
# UPDATE
###########################################################

def update():

	global t

	line = ser.readline().decode(errors="ignore").strip()

	if line == "":
		return

	try:

		gyro, acc = map(float, line.split(","))

	except ValueError:
		return

	#######################################################
	# Kalman prediction
	#######################################################

	u = np.array([[gyro]])

	kalman.predict(u)

	#######################################################
	# Kalman correction
	#######################################################

	z = np.array([[acc]])

	mu, _ = kalman.update(z)

	angle = mu[0,0]

	#######################################################
	# Store values
	#######################################################

	time_data.append(t)
	gyro_data.append(gyro)
	acc_data.append(acc)
	kalman_data.append(angle)

	gyro_curve.setData(time_data, gyro_data)
	acc_curve.setData(time_data, acc_data)
	kalman_curve.setData(time_data, kalman_data)

	t += dt


timer = QtCore.QTimer()

timer.timeout.connect(update)

timer.start(int(dt*1000))

app.exec()