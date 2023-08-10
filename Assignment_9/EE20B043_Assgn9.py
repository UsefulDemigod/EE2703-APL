from pylab import*
from mpl_toolkits.mplot3d import Axes3D

# Q1: Examples

# y=sin(sqrt(2)*t) over -pi to pi with 64 samples
t = linspace(-pi, pi, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = sin(sqrt(2)*t)
y[0] = 0         # the sample corresponding to -tmax should be set zero
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y))/64.0
w = linspace(-pi*fmax, pi*fmax, 65)[:-1]

figure(1)
subplot(2, 1, 1)
ylabel(r"$|Y|$")
title(r"Spectrum of $\sin(\sqrt{2}\times t)$")
plot(w, abs(Y), lw=2)
xlim([-10, 10])
grid()

subplot(2, 1, 2)
ylabel(r"Phase of $Y$")
xlabel(r"$\omega$")
plot(w, angle(Y), 'ro')
xlim([-10, 10])
grid()

# y=sin(sqrt(2)*t) over several time periods btw -3pi to 3pi
t1 = linspace(-pi, pi, 65)[:-1]
t2 = linspace(-3*pi, -pi, 65)[:-1]
t3 = linspace(pi, 3*pi, 65)[:-1]

figure(2)
ylabel(r"$y$")
xlabel(r"$t$")
title(r"$\sin\left(\sqrt{2}\times t\right)$")
plot(t1, sin(sqrt(2)*t1), 'b', lw=2)
plot(t2, sin(sqrt(2)*t2), 'r', lw=2)
plot(t3, sin(sqrt(2)*t3), 'r', lw=2)
grid()


# y=sin(sqrt(2)*t) with t wrapping every 2pi
t1 = linspace(-pi, pi, 65)[:-1]
t2 = linspace(-3*pi, -pi, 65)[:-1]
t3 = linspace(pi, 3*pi, 65)[:-1]
y = sin(sqrt(2)*t1)

figure(3)
ylabel(r"$y$")
xlabel(r"$t$")
title(r"$\sin\left(\sqrt{2}\times t\right)$ with $t$ wrapping every $2\pi$ ")
plot(t1, y, 'bo')
plot(t2, y, 'ro')
plot(t3, y, 'ro')
grid()


# for y=t
t = linspace(-pi, pi, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = t
y[0] = 0        # the sample corresponding to -tmax should be set zeroo
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y))/64.0  # noramlisation
w = linspace(-pi*fmax, pi*fmax, 65)[:-1]

figure(4)
semilogx(abs(w), 20*log10(abs(Y)))
xlim([1, 10])
ylim([-20, 0])
xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
ylabel(r"$|Y|$ (dB)")
title(r"Spectrum of a digital ramp")
xlabel(r"$\omega$")
grid()


# Spectrum = 0.54 + 0.46cos(2pi*n/(N-1))
t1 = linspace(-pi, pi, 65)[:-1]
t2 = linspace(-3*pi, -pi, 65)[:-1]
t3 = linspace(pi, 3*pi, 65)[:-1]
n = arange(64)
wnd = fftshift(0.54+0.46*cos(2*pi*n/63))
y = sin(sqrt(2)*t1)*wnd

figure(5)
plot(t1, y, 'bo')
plot(t2, y, 'ro')
plot(t3, y, 'ro')
ylabel(r"$y$")
xlabel(r"$t$")
title(
    r"$\sin\left(\sqrt{2}\times t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
grid()


# Spectrum of y=sin(sqrt(2)*t)*w(t) with 64 samples

t = linspace(-pi, pi, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = arange(64)
wnd = fftshift(0.54+0.46*cos(2*pi*n/63))
y = sin(sqrt(2)*t)*wnd
y[0] = 0                  # the sample corresponding to -tmax should be set zeroo
y = fftshift(y)           # make y start with y(t=0)
Y = fftshift(fft(y))/64.0  # normalisation
w = linspace(-pi*fmax, pi*fmax, 65)[:-1]

figure(6)
subplot(2, 1, 1)
plot(w, abs(Y))
xlim([-8, 8])
ylabel(r"$|Y|$")
title(r"Spectrum of $\sin\left(\sqrt{2}\times t\right)\times w(t)$")
grid()

subplot(2, 1, 2)
plot(w, angle(Y), 'ro')
xlim([-8, 8])
ylabel(r"Phase of $Y$")
xlabel(r"$\omega$")
grid()


# Spectrum of y=sin(sqrt(2)*t)*w(t) with 256 samples

t = linspace(-4*pi, 4*pi, 257)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = arange(256)
wnd = fftshift(0.54+0.46*cos(2*pi*n/256))
y = sin(sqrt(2)*t)
y = y*wnd
y[0] = 0        # the sample corresponding to -tmax should be set zeroo
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y))/256.0
w = linspace(-pi*fmax, pi*fmax, 257)[:-1]

figure(7)
subplot(2, 1, 1)
plot(w, abs(Y))
xlim([-8, 8])
ylabel(r"$|Y|$")
title(r"Spectrum of $\sin\left(\sqrt{2}\times t\right)\times w(t)$")
grid()

subplot(2, 1, 2)
plot(w, angle(Y), 'ro')
xlim([-8, 8])
ylabel(r"Phase of $Y$")
xlabel(r"$\omega$")
grid()
show()

# Q2
# Plotting a spectrum of cos^3(0.86t) with and without windowing.
y = cos(0.86*t)**3
y1 = y*wnd
y[0] = 0
y1[0] = 0
y = fftshift(y)
y1 = fftshift(y1)
Y = fftshift(fft(y))/256.0
Y1 = fftshift(fft(y1))/256.0

# spectrum of cos^3(0.86t) without windowing
figure(8)
subplot(2, 1, 1)
plot(w, abs(Y))
xlim([-4, 4])
ylabel(r"$|Y|\rightarrow$")
title(r"Spectrum of $\cos^{3}(0.86t)$ without Hamming window")
grid()
subplot(2, 1, 2)
plot(w, angle(Y), 'ro')
xlim([-4, 4])
ylabel(r"Phase of $Y\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()

# spectrum of cos^3(0.86t) with windowing
figure(9)
subplot(2, 1, 1)
plot(w, abs(Y1))
xlim([-4, 4])
ylabel(r"$|Y|\rightarrow$")
title(r"Spectrum of $\cos^{3}(0.86t)$ with Hamming window")
grid()
subplot(2, 1, 2)
plot(w, angle(Y1), 'ro')
xlim([-4, 4])
ylabel(r"Phase of $Y\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()
show()

# Q3: Find the values of w0 and delta from the spectrum of the signal.
# Let w0 = 1.5 and delta = 0.5.
w0 = 1.5
d = 0.5

t = linspace(-pi, pi, 129)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = arange(128)
wnd = fftshift(0.54+0.46*cos(2*pi*n/128))
y = cos(w0*t + d)*wnd
y[0] = 0
y = fftshift(y)
Y = fftshift(fft(y))/128.0
w = linspace(-pi*fmax, pi*fmax, 129)
w = w[:-1]

figure(10)
subplot(2, 1, 1)
plot(w, abs(Y))
xlim([-4, 4])
ylabel(r"$|Y|\rightarrow$")
title(r"Spectrum of $\cos(w_0t+\delta)$ with Hamming window")
grid()
subplot(2, 1, 2)
plot(w, angle(Y), 'ro')
xlim([-4, 4])
ylabel(r"Phase of $Y\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()
show()
# w0 is calculated by finding the weighted average of all w>0. Delta is found by calculating the phase at w closest to w0.
ii = where(w >= 0)
wcal = sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2)
i = abs(w-wcal).argmin()
delta = angle(Y[i])
print("Calculated value of w0 without noise: ", wcal)
print("Calculated value of delta without noise: ", delta)

# Q4: Repeating the above process for noisy signals
y = (cos(w0*t + d) + 0.1*randn(128))*wnd
y[0] = 0
y = fftshift(y)
Y = fftshift(fft(y))/128.0

figure(11)
subplot(2, 1, 1)
plot(w, abs(Y))
xlim([-4, 4])
ylabel(r"$|Y|\rightarrow$")
title(r"Spectrum of a noisy $\cos(w_0t+\delta)$ with Hamming window")
grid()
subplot(2, 1, 2)
plot(w, angle(Y), 'ro')
xlim([-4, 4])
ylabel(r"Phase of $Y\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()
show()

# w0 is calculated by finding the weighted average of all w>0. Delta is found by calculating the phase at w closest to w0.
ii = where(w >= 0)
wcal = sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2)
i = abs(w-wcal).argmin()
delta = angle(Y[i])
print("Calculated value of w0 with noise: ", wcal)
print("Calculated value of delta with noise: ", delta)

# Q5
# We have to plot the spectrum of a chirped signal.
t = linspace(-pi, pi, 1025)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = arange(1024)
wnd = fftshift(0.54+0.46*cos(2*pi*n/1024))
y = cos(16*t*(1.5 + t/(2*pi)))*wnd
y[0] = 0
y = fftshift(y)
Y = fftshift(fft(y))/1024.0
w = linspace(-pi*fmax, pi*fmax, 1025)
w = w[:-1]

figure(12)
subplot(2, 1, 1)
plot(w, abs(Y))
xlim([-100, 100])
ylabel(r"$|Y|\rightarrow$")
title(r"Spectrum of chirped function")
grid()
subplot(2, 1, 2)
plot(w, angle(Y), 'ro')
xlim([-100, 100])
ylabel(r"Phase of $Y\rightarrow$")
xlabel(r"$\omega\rightarrow$")
grid()
show()

# Q6:
# We have to plot a surface plot with respect to t and w.
Y_mag = zeros((16, 64))
Y_phase = zeros((16, 64))
t = linspace(-pi, pi, 1025)[:-1]
t_array = split(t, 16)

for i in range(len(t_array)):
    n = arange(64)
    wnd = fftshift(0.54+0.46*cos(2*pi*n/64))
    y = cos(16*t_array[i]*(1.5 + t_array[i]/(2*pi)))*wnd
    y[0] = 0
    y = fftshift(y)
    Y = fftshift(fft(y))/64.0
    Y_mag[i] = abs(Y)
    Y_phase[i] = angle(Y)

t = t[::64]
w = linspace(-fmax*pi, fmax*pi, 64+1)[:-1]

t, w = meshgrid(t, w)

fig1 = figure(13)
ax = fig1.add_subplot(111, projection='3d')
surf = ax.plot_surface(w, t, Y_mag.T, cmap='viridis',
                       linewidth=0, antialiased=False)
fig1.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title('surface plot')
ylabel(r"$\omega\rightarrow$")
xlabel(r"$t\rightarrow$")
show()
