"""
        			EE2703:Applied Programming Lab
      				    Assignment 7: Solution
Name  :Harisankar K J
RollNo:EE20B043
"""
# Importing Libraries
from sympy import *
import numpy as np
import scipy.signal as sp
from pylab import *

s = symbols('s')
t = symbols('t')


def lowpass(R1, R2, C1, C2, G, Vi):
    s = symbols('s')
    A = Matrix([[0, 0, 1, -1/G],
                [-1/(1+s*R2*C2), 1, 0, 0],
                [0, -G, G, 1],
                [-1/R1-1/R2-s*C1, 1/R2, 0, s*C1]])
    b = Matrix([0, 0, 0, -Vi/R1])
    V = A.inv()*b
    return V

# Transfer Function


V = lowpass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
Vo = V[3]

# Bode plot of H(s)

figure(1)
w = logspace(0, 8, 801)
ss = 1j*w
hf = lambdify(s, Vo, 'numpy')
v = hf(ss)
loglog(w, abs(v), lw=2)
title(r"$Magnitude (|H(j\omega))| Response$ for lowpass filter")
xlabel(r'$\omega\rightarrow$')
ylabel(r'$|H(j\omega)|\rightarrow$')
grid()


# Function to convert sympy function into a transfer function.
def sympytoscipy(Y):
    n, d = fraction(simplify(Y))
    num, den = (np.array(Poly(n, s).all_coeffs(), dtype=float),
                np.array(Poly(d, s).all_coeffs(), dtype=float))
    H = sp.lti(num, den)
    return H


H = sympytoscipy(Vo)  # Transfer Function

t, y1 = sp.impulse(H, None, linspace(0, 5e-3, 100000))  # Impulse Response

# Q1: Using Step Response

V = lowpass(10000, 10000, 1e-9, 1e-9, 1.586, 1/s)
Vo = V[3].simplify()

H1 = sympytoscipy(Vo)
t, y1 = sp.impulse(H1, None, linspace(0, 5e-3, 100000))
figure(2)
plot(t, y1)
title(r"Step Response for low pass filter")
xlabel(r'$t\rightarrow$')
ylabel(r'$V_o(t)\rightarrow$')
grid()

# Q2: Response for sum of sinusoids

v_in = sin(2000*pi*t) + cos(2e6*pi*t)
t, y, svec = sp.lsim(H, v_in, t)

# The plot for output response for sum of sinusoids of a lowpass filter.
figure(3)
plot(t, v_in, 'r', label='input')
plot(t, y, 'b', label='output')
title(r"Input and Output voltage for sum of sinusoids")
xlabel(r'$t\rightarrow$')
ylabel(r'$V(t)\rightarrow$')
legend()
grid()

# Q3 High pass filter


def highpass(R1, R3, C1, C2, G, Vi):
    A = Matrix([[0, -1, 0, 1/G],
                [s*C2*R3/(s*C2*R3+1), 0, -1, 0],
                [0, G, -G, 1],
                [-1*s*C2-1/R1-s*C1, 0, s*C2, 1/R1]])
    b = Matrix([0, 0, 0, -Vi*s*C1])
    V = A.inv()*b
    return V


V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
Vo = V[3].simplify()

# Plotting the bode plot of H(s)
figure(4)
w = logspace(0, 8, 801)
s = symbols('s')
ss = 1j*w
hf = lambdify(s, Vo, 'numpy')
v = hf(ss)
loglog(w, abs(v), lw=2)
title(r"$Magnitude (|H(j\omega))| Response$ for highpass filter")
xlabel(r'$\omega\rightarrow$')
ylabel(r'$|H(j\omega)|\rightarrow$')
grid()


H = sympytoscipy(Vo)
# Q5: Step Response of High pass filter

V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1/s)

Vo = V[3].simplify()
H1 = sympytoscipy(Vo)

t, y1 = sp.impulse(H1, None, linspace(0, 5e-3, 100000))
figure(5)
plot(t, y1)
title(r"Step Response for High pass filter")
xlabel(r'$t\rightarrow$')
ylabel(r'$V_o(t)\rightarrow$')
grid()

# Q4: High pass filter to Damped Sinusoids

# High frequency

t = linspace(0, 1e-5, 100000)
v_in = exp(-5000*t)*sin(2e6*pi*t)
t, y, svec = sp.lsim(H, v_in, t)
figure(6)
plot(t, v_in, 'r', label='input')
plot(t, y, 'b', label='output')
title(r"Input and Output voltage for damped high frequency sinusoid")
xlabel(r'$t\rightarrow$')
ylabel(r'$V(t)\rightarrow$')
legend()
grid()

# Low frequency

t = linspace(0, 1e-2, 10000)
v_in = exp(-500*t)*sin(2e3*pi*t)*heaviside(0, 1)
t, y, svec = sp.lsim(H, v_in, t)
figure(7)
plot(t, v_in, 'r', label='input')
plot(t, y, 'b', label='output')
title(r"Input and Output voltage for damped low frequency sinusoid")
xlabel(r'$t\rightarrow$')
ylabel(r'$V(t)\rightarrow$')
legend()
grid()
show()
