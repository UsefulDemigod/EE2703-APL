"""
        			EE2703:Applied Programming Lab
      				    Assignment 5: Solution
Name  :Harisankar K J
RollNo:EE20B043
"""
import sys
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.pyplot as plt

Nx = 25       # size along x
Ny = 25       # size along y
radius = 8    # radius of central lead
Niter = 1500  # number of iterations to perform
n = arange(Niter)

if len(sys.argv) == 5:
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
    radius = int(sys.argv[3])
    Niter = int(sys.argv[4])

# Allocating potential array and initalizing it
phi = np.zeros((Ny, Nx), dtype=float)
x = np.linspace(-0.5, 0.5, Nx)
y = np.linspace(-0.5, 0.5, Ny)
Y, X = np.meshgrid(y, x)
volt1Nodes = np.where(X**2+Y**2 <= 0.35**2)
phi[volt1Nodes] = 1.0

# plotting coutours fot the phi values
plt.figure(1)
plt.xlabel("x")
plt.ylabel("y")
plt.contourf(range(Nx), range(Ny), phi)
plt.colorbar()
plt.grid()
plt.title("Contour plot of potential($\phi$) before iteration")
plt.show()

errors = np.zeros(Niter)
for k in range(Niter):

    # saving the old value of phi
    oldphi = phi.copy()

    phi[1:-1, 1:-1] = 0.25*(oldphi[1:-1, 0:-2] + oldphi[1:-1,
                            2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])  # updating phi

    # boundary conditions
    phi[0, 1:-1] = 0               # bottom edge
    phi[1:-1, 0] = phi[1:-1, 1]    # left edge
    phi[-1, 1:-1] = phi[-2, 1:-1]  # top edge
    phi[1:-1, -1] = phi[1:-1, -2]  # right edge

    phi[0, 0] = 0.5*(phi[0, 1] + phi[1, 0])  # updatig the left out corners
    phi[0, -1] = 0.5*(phi[0, -2] + phi[1, -1])
    phi[-1, 0] = 0.5*(phi[-1, 1] + phi[-2, 0])
    phi[-1, -1] = 0.5*(phi[-1, -2] + phi[-2, -1])

    # resetting the voltage = 1 in region in contact with electrode
    phi[volt1Nodes] = 1.0

    # maxmum error after each ieration is stored
    errors[k] = (abs(phi-oldphi)).max()

# print(errors)

# using least squares method to apprximate the errors wrt iterations
""""
y = A*exp(Bx) or
logy = logA + B*x
M = [1 x]
p = [p1 p0]T
Here our parameters p0 and p1 are logA and B respectively 
"""
iterations = np.asarray(range(Niter))
iter50 = iterations[::50]
logerr = np.log(errors[::50])

M = np.c_[iter50, np.ones(Niter//50)]
p = np.linalg.lstsq(M, logerr, rcond=None)[0]
prederror = np.exp(p[1]+p[0]*iter50)

M500 = np.c_[iterations[500::50], np.ones((Niter - 500)//50)]
p500 = np.linalg.lstsq(M500, logerr[10::])[0]
prederror500 = np.exp(p500[1]+p500[0]*iter50[10:])

# the current density at each point is calculated by using the formula given
Jx = zeros((Ny, Nx))
Jy = zeros((Ny, Nx))
Jx, Jy = (1/2*(phi[1:-1, 0:-2]-phi[1:-1, 2:]),
          1/2*(phi[:-2, 1:-1]-phi[2:, 1:-1]))
print(size(Jx), "    ", size(Jy))
# Plotting of initial potential contour.
plt.figure(2)
plt.title("Initial Potential Contour")
plt.xlabel(r'$X\rightarrow$')
plt.ylabel(r'$Y\rightarrow$')
plt.plot(volt1Nodes[0]/Nx-0.48, volt1Nodes[1]/Ny-0.48, 'ro', label="V = 1")
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.grid(True)
plt.legend()
plt.show()

# Plotting the value of error vs iteration in semilog.
plt.figure(3)
plt.title("Error versus iteration")
plt.xlabel(r'$No. Of Iterations\rightarrow$')
plt.ylabel(r'$Error\rightarrow$')
plt.semilogy(range(Niter), errors)
plt.grid(True)
plt.show()

# Plotting the value of error vs iteration in loglog.
plt.figure(4)
plt.title("Error with each iteration in loglog")
plt.xlabel(r'$No. of Iterations\rightarrow$')
plt.ylabel(r'$Error\rightarrow$')
plt.loglog(n, errors)
plt.grid(True)
plt.show()

# Plotting the value of error vs iteration above 500 in semilog .
plt.figure(5)
plt.title("Error versus iteration above 500 in semilog")
plt.xlabel(r'$No. of Iterations\rightarrow$')
plt.ylabel(r'$Error\rightarrow$')
plt.semilogy(arange(Niter)[500:], errors[500:])
plt.grid(True)
plt.show()

# Plotting the value of error vs iteration above 500 in loglog .
plt.figure(6)
plt.title("Error versus iteration above 500 in loglog")
plt.xlabel(r'$No. of Iterations\rightarrow$')
plt.ylabel(r'$Error\rightarrow$')
plt.loglog(arange(Niter)[500:], errors[500:])
plt.grid(True)
plt.show()

# Plotting of actual and expected error (above 500 iterations) in semilog.
plt.figure(7)
plt.semilogy(arange(Niter)[500:], errors[500:], label="True Value")
plt.semilogy(arange(Niter)[500::50], np.exp(
    p500[1])*exp(p500[0]*arange(Niter)[500::50]), "bo", label="Expected Value")
plt.title("Expected versus actual error (>500 iterations) in semilog")
plt.xlabel(r'$Iteration\rightarrow$')
plt.ylabel(r'$Error\rightarrow$')
plt.grid(True)
plt.legend()
plt.show()

# Plotting of actual and expected error in semilog.
plt.figure(8)
plt.semilogy(arange(Niter), errors, label="True Value")
plt.semilogy(arange(Niter)[::50], np.exp(
    p[1])*exp(p[0]*arange(Niter)[::50]), "go", label="Expected Value")
plt.title("Expected versus actual error in semilog")
plt.xlabel(r'$Iteration\rightarrow$')
plt.ylabel(r'$Error\rightarrow$')
plt.grid(True)
plt.legend()
plt.show()

# Plotting of the contour of phi (potential).
plt.figure(9)
plt.contourf(Y, X, phi)
plt.plot(volt1Nodes[0]/Nx-0.48, volt1Nodes[1]/Ny-0.48, 'ro', label="V = 1")
plt.title("Contour plot of potential")
plt.xlabel(r'$X\rightarrow$')
plt.ylabel(r'$Y\rightarrow$')
plt.colorbar()
plt.grid(True)
plt.legend()
plt.show()

# plotting of the current vector plot along with the potential.
plt.figure(10)
plt.quiver(-Y[1:-1, 1:-1], X[1:-1, 1:-1], Jx[:, ::-1], Jy)
plt.title("The vector plot of the current flow")
plt.xlabel(r'$X\rightarrow$')
plt.ylabel(r'$Y\rightarrow$')
plt.plot(volt1Nodes[0]/Nx-0.48, volt1Nodes[1]/Ny-0.48, 'ro')
plt.show()

# Plotting the surface plots of phi.
fig10 = figure(11)
ax = p3.Axes3D(fig10)
title("The 3-D surface plot of the potential")
xlabel(r'$X\rightarrow$')
ylabel(r'$Y\rightarrow$')
surf = ax.plot_surface(-X, -Y, phi, rstride=1, cstride=1, cmap=cm.jet)
fig10.colorbar(surf)
show()
