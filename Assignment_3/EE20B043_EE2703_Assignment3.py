"""
        			EE2703:Applied Programming Lab
      				    Assignment 3: Solution
Name  :Harisankar K J
RollNo:EE20B043
"""
from pylab import *
import scipy.special as sp
import scipy
import numpy as np

# Q2: Loading the data into a matrix
yf = np.loadtxt("fitting.dat")
(N, k) = yf.shape
t = yf[:, 0]

# Q3: all columns
# standard deviation
scl = logspace(-1, -3, 9)
figure(0)
for i in range(1, k):
    plot(t, yf[:, i], label=("sigma = " + str(round(scl[i-1], 4))))
xlabel(r'$t$', size=20)
ylabel(r'$f(t)+n$', size=20)
title(r'Plot of the data to be fitted')

# Q4 : plotting the true value function


def g(tk=t, A=1.05, B=-0.105):
    return A*sp.jn(2, tk)+B*t


y = g()
plot(t, y, label='true')
legend()
grid(True)
show()

# Q5: plotting the first column data values using errorbars
figure(1)
plot(t, y, label='true')
errorbar(t[::5], yf[:, 1][::5], scl[0], fmt='ro')
xlabel('$t\\rightarrow$')
title('Data Points with Error for St.Dev=0.10')
grid(True)
show()

# Q6: creating matrix equation
y1 = sp.jn(2, t)
M = c_[y1, t]
AB = array((1.05, -0.105))
if allclose(g(), dot(M, AB)):
    print("The values of the matrices match")
else:
    print("matrix values no  similar")

# Q7: computing mean squared error
n = 21
A = linspace(0, 2, n)
B = linspace(-0.2, 0, n)
eps = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        eps[i][j] = mean(square(yf[:, 1]-g(t, A[i], B[j])))

# print(eps)

# 8: plotting contours
figure(2)
pl = contour(A, B, eps, levels=20)
xlabel('$A\\rightarrow$')
ylabel('$B\\rightarrow$')
title(r'Contour Plot')
clabel(pl)
grid(True)
show()

# 9: fitting data by least squares
ex = np.zeros((2, 1))
ex = scipy.linalg.lstsq(M, y)[0]

# 10: error in the estimate
fit = np.zeros((k-1, 2))
for i in range(k-1):
    fit[i] = scipy.linalg.lstsq(M, yf[:, i+1])[0]
Ae = np.zeros((k-1, 1))
Be = np.zeros((k-1, 1))
for i in range(k-1):
    Ae[i] = square(fit[i][0]-ex[0])
    Be[i] = square(fit[i][1]-ex[1])
figure(3)
plot(scl, Ae, label='A')
plot(scl, Be, label='B')
xlabel('Noise standard deviation')
ylabel('MS error')
title('Error in Estimate')
legend()
grid(True)
show()
#print(Ae[:, 0])
# Q11: replotting error in estimate using loglog
figure(4)
loglog(scl, Ae[:, 0], 'ro', label='A error(in logscale)')
loglog(scl, Be[:, 0], 'go', label='B error(in logscale)')
errorbar(scl, Ae[:, 0], std(Ae[:, 0]), fmt='ro')
errorbar(scl, Be[:, 0], std(Be[:, 0]), fmt='go')
xlabel('Noise Standard Deviation')
ylabel('MS Error (in logscale)')
title('Error in Estimate (in logscale)')
legend()
grid(True)
show()
