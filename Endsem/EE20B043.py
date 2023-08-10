"""
        			EE2703:Applied Programming Lab
      				          Endsem
Name  :Harisankar K J
RollNo:EE20B043
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

# initializing values
l = 0.5
N = 4  # change the value of N from here to 100 if needed
dz = l/N
lmda = 2.0
k = 2*np.pi/lmda
a = 0.01
mu0 = 4e-7*np.pi

# Q1
i = np.linspace(-N, N, 2*N+1)  # i ranges from -N to N
z = i*dz                       # Defining the vector z
I = np.array
Im = 1.0
Iupper = Im*np.sin(k*(l - z[N:]))
Ilower = Im*np.sin(k*(l + z[:-(N+1)]))
I = np.append(Ilower, Iupper)            # I calculated from formula
u = np.delete(z, 0)
u = np.delete(u, -1)
u = np.delete(u, N-1)
j = np.delete(I, 0)
j = np.delete(j, -1)
j = np.delete(j, N-1)

# Q2
Id = np.identity(2*N-2)
M = 1/(2*np.pi*a)*Id

# Q3
Rz = np.sqrt(np.add(a*a, np.square(np.subtract.outer(z, z))))
Ru = np.sqrt(np.add(a*a, np.square(np.subtract.outer(u, u))))
list = np.delete(np.arange(1, 2*N), N-1)
RiN = Rz[list, N]
Pb = (mu0/(4*np.pi))*((np.exp(-1j*k*RiN))*dz)/RiN
P = (mu0/(4*np.pi))*((np.exp(-1j*Ru*k))/Ru)*dz

# Q4:
Qb = -(Pb*a/mu0*(-1j*k/RiN-1/RiN**2))
Q = -(P*a/mu0*(-1j*k/Ru-1/Ru**2))

# Q5
J = np.dot(np.linalg.inv(M-Q), (Qb*Im))
list = J.tolist()                          # Considering Boundary conditions
list.insert(0, 0)
list.insert(N, Im)
list.insert(2*N, 0)
J = np.array(list)

# Plotting
plt.figure()
plt.title("I vs z for N = "+str(N))
plt.xlabel("z")
plt.ylabel("I")
plt.plot(z, I, label="I from formula")
plt.plot(z, J, label="I exact")
plt.grid()
plt.legend()
plt.show()

# Printing Values
print("\nz = \n", z.round(2))
print("\nRu= \n", Ru.round(2))
print("\nRz= \n", Rz.round(2))
print("\nRiN = \n", Rz[N, :].round(2))
print("\nPb= \n", (Pb*1e8).round(2))
print("\nP= \n", (P*1e8).round(2))
print("\nQb = \n", Qb.round(2))
print("\nI = \n", I.round(2))
print("\nJ = \n", J.round(8))
