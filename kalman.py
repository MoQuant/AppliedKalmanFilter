import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

bg = 'white'
fg = 'black'

fig = plt.figure(figsize=(10, 6))
price = fig.add_subplot(121)
error = fig.add_subplot(122)

fig.patch.set_facecolor(bg)
price.set_facecolor(bg)
error.set_facecolor(bg)

inset_ax = inset_axes(error, width="20%", height="20%", loc='upper right')
inset_ax.set_facecolor(bg)

for i in ('x', 'y'):
    price.tick_params(i, colors=fg)
    error.tick_params(i, colors=fg)
    inset_ax.tick_params(i, colors=fg)

V = lambda m, n: np.array([[0 for j in range(n)] for i in range(m)])

def Store(i, sn, x, y):
    if i < len(sn):
        sn[i][0] = x
        sn[i][1] = y
    else:
        for k in range(1, len(sn)):
            sn[k-1][0] = sn[k][0]
            sn[k-1][1] = sn[k][1]
        sn[len(sn) - 1][0] = x
        sn[len(sn) - 1][1] = y
    return sn

def XK1(xk, a):
    FK = np.array([[1, 1], [0, 1]])
    u = np.array([[0.5], [1.0]])
    return FK.dot(xk) + a*u

def PK1(Pk, Q):
    FK = np.array([[1, 1], [0, 1]])
    return FK.dot(Pk.dot(FK.T)) + Q

def COV(x):
    m, n = x.shape
    mu = (1/m)*np.ones(m).dot(x)
    cov = (1/(m-1))*(x - mu).T.dot(x - mu)
    return cov


LIMIT = 252
lookback = 20

ticker = 'SPY'
data = pd.read_csv(f'{ticker}.csv')

dates = data['date'].values
position = data['adjClose'].values
velocity = data['changeOverTime'].values
acceleration = velocity[:-1] - velocity[1:]

dates = dates[:-1]
position = position[:-1]
velocity = velocity[:-1]

dates = dates[::-1][-LIMIT:]
position = position[::-1][-LIMIT:]
velocity = velocity[::-1][-LIMIT:]
acceleration = acceleration[::-1][-LIMIT:]


# Delcare Variables

xk1 = V(2, 1)
pk1 = V(2, 2)
K1 = V(2, 2)
x1k = V(2, 1)
p1k = V(2, 2)
H = np.eye(2)
Q = np.array([[rd.random(), rd.random()], [rd.random(), rd.random()]])
R = Q

zk = V(2, 1)
xk = V(2, 1)
pk = V(2, 2)

store_noise = V(lookback, 2)
store_z = V(lookback, 2)

X = np.array([[position[0]], [velocity[0]]])

rx, ry, sy = [], [], []
dx, se = [], []

plt.pause(3)
for i in range(1, LIMIT):
    a = acceleration[i - 1]

    # Prediction Function (xk1)
    xk1 = XK1(X, a) 
    zk[0][0] = position[i]
    zk[1][0] = velocity[i]

    store_noise = Store(i, store_noise, abs(zk[0][0] - xk1[0][0]), abs(zk[1][0] - xk1[1][0]))
    store_z = Store(i, store_z, zk[0][0], zk[1][0])

    if i > 1:
        Q = COV(store_noise)
        R = COV(store_z)

    # Prediction Function (pk1)
    pk1 = PK1(pk, Q)

    # Update estimate
    uu = pk1.dot(H.T)
    vv = H.dot(uu)
    ww = vv + R

    # Kalman Gain
    K1 = uu.dot(np.linalg.inv(ww))

    y = zk - H.dot(xk1)
    th = K1.dot(y)

    x1k = xk1 + th
    p1k = pk1 - K1.dot(H.dot(xk1))

    #print(xk1[0][0], zk[0][0])

    # PLOT RESULTS
    price.cla()
    error.cla()
    inset_ax.cla()

    rx.append(i)
    ry.append(position[i])
    sy.append(xk1[0][0])
    se.append(abs(position[i] - xk1[0][0]))

    price.set_title("Stock Price Estimates")
    price.set_xlabel("Time")
    price.set_ylabel("Stock Price")

    error.set_title("Dollar Error")
    error.set_xlabel("Time")
    error.set_ylabel("Error")

    price.plot(rx, ry, color='red', label='Actual Price')
    price.plot(rx, sy, color='limegreen', label='Predicted Price')

    error.hist(se, bins=10, color='blue')

    inset_ax.plot(rx[-10:], ry[-10:], color='red')
    inset_ax.plot(rx[-10:], sy[-10:], color='limegreen')
    price.legend()

    plt.pause(0.01)
    
    

    X = x1k
    pk = p1k















plt.show()
