"""
        			EE2703:Applied Programming Lab
      				    Assignment 8: Solution
Name  :Harisankar K J
RollNo:EE20B043
"""
from pylab import *

# Q1: Working through the examples

# magnitude and phase plot for the DFT of sin(5t)

N1 = 128
t1 = linspace(-pi, pi, N1+1)
t1 = t1[:-1]
y1 = sin(5*t1)
Y1 = fftshift(fft(y1))/N1
w1 = linspace(-64, 63, N1)

figure(1)
plot(w1, abs(Y1))
xlim([-10, 10])
title(r"Spectrum of $\sin(5t)$")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()

figure(2)
plot(w1, angle(Y1), 'ro')
ii = where(abs(Y1) > 1e-3)
plot(w1[ii], angle(Y1[ii]), 'go')
xlim([-10, 10])
title(r"Phase of $\sin(5t)$")
ylabel(r"$\angle Y(\omega)\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()
show()

# magnitude and phase plot for the DFT of (1 + 0.1cos(t))cos(10t)

N2 = 512
t2 = linspace(-4*pi, 4*pi, N2+1)
t2 = t2[:-1]
y2 = (1 + 0.1*cos(t2))*cos(10*t2)
Y2 = fftshift(fft(y2))/N2
w2 = linspace(-64, 64, N2+1)
w2 = w2[:-1]

figure(3)
plot(w2, abs(Y2))
xlim([-15, 15])
title(r"Spectrum of $(1 + 0.1cos(t))cos(10t)$")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()

figure(4)
plot(w2, angle(Y2), 'ro')
xlim([-15, 15])
title(r"Phase of $(1 + 0.1cos(t))cos(10t)$")
ylabel(r"$\angle Y(\omega)\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()
show()

# Q2:
# magnitude and phase plot for the DFT of sin^3t
N3 = 512
t3 = linspace(-4*pi, 4*pi, N3+1)
t3 = t3[:-1]
y3 = (3*sin(t3) - sin(3*t3))/4
Y3 = fftshift(fft(y3))/N3
w3 = linspace(-64, 64, N3+1)
w3 = w3[:-1]

figure(5)
plot(w3, abs(Y3))
xlim([-5, 5])
title(r"Spectrum of $sin^3(t)$")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()

figure(6)
plot(w3, angle(Y3), 'ro')
xlim([-5, 5])
title(r"Phase of $sin^3(t)$")
ylabel(r"$\angle Y(\omega)\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()
show()

# magnitude and phase plot for the DFT of cos^3t
N4 = 512
t4 = linspace(-4*pi, 4*pi, N4+1)
t4 = t4[:-1]
y4 = (3*cos(t3) + cos(3*t3))/4
Y4 = fftshift(fft(y4))/N4
w4 = linspace(-64, 64, N4+1)
w4 = w4[:-1]

figure(7)
plot(w4, abs(Y4))
xlim([-5, 5])
title(r"Spectrum of $cos^3(t)$")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()

figure(8)
plot(w4, angle(Y4), 'ro')
xlim([-5, 5])
title(r"Phase of $cos^3(t)$")
ylabel(r"$\angle Y(\omega)\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()
show()

# Q3: magnitude and phase plot for the DFT of cos(20t + 5cos(t))
y5 = cos(20*t3 + 5*cos(t3))
Y5 = fftshift(fft(y5))/N3

figure(9)
plot(w3, abs(Y5))
xlim([-40, 40])
title(r"Spectrum of $cos(20t + 5cos(t))$")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid(True)

figure(11)
ii = where(abs(Y5) >= 1e-3)
plot(w3[ii], angle(Y5[ii]), 'go')
xlim([-40, 40])
title(r"Phase of $cos(20t + 5cos(t))$")
ylabel(r"$\angle Y(\omega)\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid(True)
show()

# Q4: Finding the max error in the magnitude of calculated DTF of exp(-0.5t^2)
N = 128
T = 2*pi
maxerror = 1e-6
n = 0
error = 0
w = 0
Y = 0
while error < maxerror:

    t = linspace(-T/2, T/2, N+1)[:-1]
    w = N/T * linspace(-pi, pi, N+1)[:-1]
    y = exp(-0.5*t**2)

    Y_True = (1/sqrt(2*pi))*exp(-0.5*w**2)
    Y = fftshift(fft(y))*T/(2*pi*N)
    error = max(abs(abs(Y)-Y_True))

    T = T*2
    N = N*2

    n = n+1

print(" Max Error: {} \n No. of Interations: {}".format(error, n))
print(" Value for T: {}*pi \n Value for N: {}" .format(T/pi, N))

# magnitude plots for different values of N and Ts
y = exp(-0.5*t1**2)
Y = fftshift(fft(y))/N1
figure(12)
plot(w1, abs(Y))
title(r"Spectrum of a Gaussian function")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid(True)
xlim([-10, 10])

y = exp(-0.5*t2**2)
Y = fftshift(fft(y))/N2
plot(w2, abs(Y))
title(r"Spectrum of a Gaussian function")
ylabel(r"$|Y(\omega)|\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid(True)
xlim([-10, 10])
show()
