"""
        			EE2703:Applied Programming Lab
      				    Assignment 6: Solution
Name  :Harisankar K J
RollNo:EE20B043
"""
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Q1
# defining and finding inverse laplace.
num1 = np.poly1d([1, 0.5])
den1 = np.polymul([1, 1, 2.5], [1, 0, 2.25])
X1 = sp.lti(num1, den1)
t1, x1 = sp.impulse(X1, None, np.linspace(0, 50, 500))

# Plotting x(t) from 0-50 seconds
plt.figure(1)
plt.plot(t1, x1)
plt.title("x(t) vs t for decay of 0.5")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$x(t)\rightarrow$')
plt.grid()
plt.show()

# Q2
# defining and finding inverse laplace.
num2 = np.poly1d([1, 0.05])
den2 = np.polymul([1, 0.1, 2.2525], [1, 0, 2.25])
X2 = sp.lti(num2, den2)
t2, x2 = sp.impulse(X2, None, np.linspace(0, 50, 500))

# Plotting x(t) from 0-50 seconds
plt.figure(2)
plt.plot(t2, x2)
plt.title("x(t) vs t for decay of 0.05")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$x(t)\rightarrow$')
plt.grid()
plt.show()

# Q3
# definig the transfer function and plotting the convolution values for different w
H = sp.lti([1], [1, 0, 2.25])
for w in np.arange(1.4, 1.6, 0.05):
    t = np.linspace(0, 50, 500)
    f = np.cos(w*t)*np.exp(-0.05*t)
    t, x, svec = sp.lsim(H, f, t)

    plt.figure(3)  # plotting the value for each w
    plt.plot(t, x, label='w = ' + str(w))
    plt.title("x(t) for different frequencies")
    plt.xlabel(r'$t\rightarrow$')
    plt.ylabel(r'$x(t)\rightarrow$')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

# Q4
# soling for X and Y in the coupling equation and plotting the y(t) and x(t) vs t graph
X3 = sp.lti([1, 0, 2], [1, 0, 3, 0])
Y3 = sp.lti([2], [1, 0, 3, 0])
t3, x3 = sp.impulse(X3, None, np.linspace(0, 20, 5001))
t3, y3 = sp.impulse(Y3, None, np.linspace(0, 20, 5001))

plt.figure(4)
plt.plot(t3, x3, label='x(t)')
plt.plot(t3, y3, label='y(t)')
plt.title("values of x(t) and y(t)")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$functions\rightarrow$')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# Q5
# finding the transfer function of the rlc circuit and plotting bode
den = np.poly1d([1e-12, 1e-4, 1])
Hrlc = sp.lti([1], den)
w, S, phi = Hrlc.bode()

plt.figure(5)  # magnitude bode plot
plt.semilogx(w, S)
plt.title("Magnitude Bode Plot")
plt.xlabel(r'$w\rightarrow$')
plt.ylabel(r'$20\log|H(jw)|\rightarrow$')
plt.grid()

plt.figure(6)  # phase bode plot
plt.semilogx(w, phi)
plt.title("Phase Bode Plot")
plt.xlabel(r'$\omega\rightarrow$')
plt.ylabel(r'$\angle H(j\omega)\rightarrow$')
plt.grid(True)
plt.show()

# Q6
# finding the value of Vout and plotting for different ranges
t4 = np.linspace(0, 30e-6, 10000)
vi1 = np.cos(1e3*t4) - np.cos(1e6*t4)
t4, vo1, svec = sp.lsim(Hrlc, vi1, t4)

plt.figure(7)  # for large intervals
plt.plot(t4, vo1)
plt.title("Vout for t<30us")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$V_o(t)\rightarrow$')
plt.grid()

t5 = np.linspace(0, 30e-3, 10000)
vi2 = np.cos(1e3*t5) - np.cos(1e6*t5)
t5, vo2, svec = sp.lsim(Hrlc, vi2, t5)

plt.figure(8)
plt.plot(t5, vo2)
plt.title("Vout for t<30ms")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$V_o(t)\rightarrow$')
plt.grid()
plt.show()
