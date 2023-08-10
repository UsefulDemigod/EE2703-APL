"""
        			EE2703:Applied Programming Lab
      				    Assignment 4: Solution
Name  :Harisankar K J
RollNo:EE20B043
"""
from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.integrate import quad
import scipy as sp

pi = np.pi


def exp(x):
    return np.exp(x)


def coscos(x):
    return np.cos(np.cos(x))


x = np.arange(-2*pi, 4*pi, 0.01)
# x_fourier= np.arange(0,2*pi,0.01) #if we want to restrict the graphs
# print(x)
def fc_exp(x): return exp(x)*cos(i*x)  # i dummy index
def fs_exp(x): return exp(x)*sin(i*x)


n = 26  # max value of I, not taken infinity, better result with high value

Anexp = []  # defining array

Bnexp = []

sumexp = 0

for i in range(n):
    an = quad(fc_exp, 0, 2*pi)[0]*(1.0/np.pi)
    Anexp.append(an)

for i in range(n):
    bn = quad(fs_exp, 0, 2*pi)[0]*(1.0/np.pi)
    Bnexp.append(bn)  # putting value in array Bn

for i in range(n):
    if i == 0.0:
        sumexp = sumexp+Anexp[i]/2
    else:
        sumexp = sumexp+(Anexp[i]*np.cos(i*x)+Bnexp[i]*np.sin(i*x))


def fc_coscos(x): return coscos(x)*cos(i*x)  # i dummy index
def fs_coscos(x): return coscos(x)*sin(i*x)


Ancoscos = []  # defining array

Bncoscos = []

sumcoscos = 0

for i in range(n):
    an1 = quad(fc_coscos, 0, 2*pi)[0]*(1.0/np.pi)
    Ancoscos.append(an1)

for i in range(n):
    bn1 = quad(fs_coscos, 0, 2*pi)[0]*(1.0/np.pi)
    Bncoscos.append(bn1)  # putting value in array Bn

for i in range(n):
    if i == 0.0:
        sumcoscos = sumcoscos+Ancoscos[i]/2
    else:
        sumcoscos = sumcoscos+(Ancoscos[i]*np.cos(i*x)+Bncoscos[i]*np.sin(i*x))

# print(Anexp)
plt.figure(1)
plt.semilogy(x, exp(x))
plt.semilogy(x, sumexp, 'og')
plt.title(r'$e^x$ on a semilogy plot')
plt.xlabel('x')
plt.ylabel(r'$log(e^x)$')
plt.grid()
plt.show()

plt.figure(2)
plt.plot(x, coscos(x))
plt.plot(x, sumcoscos, 'og')
plt.title(r'Plot of $cos(cos(x))$')
plt.xlabel('x')
plt.ylabel(r'$cos(cos(x))$')
plt.grid()
plt.show()
# print(len(Bnexp))
# print(Bnexp[0])
# print(len(np.arange(0, n, 1)))
# print(Bnexp)
plt.figure(3)
plt.semilogy(np.arange(0, n, 1), np.abs(Anexp),
             "ro", label="Coeff. An and Bn Exp")
plt.semilogy(np.arange(0, n, 1), np.abs(Bnexp), "ro")
plt.title(r"Coeff. of fourier series of $e^x$ on a semilogy scale")
plt.xlabel(r'$n$')
plt.ylabel(r'$log(coeff)$')
plt.legend()
plt.grid()
plt.show()

plt.figure(4)
plt.loglog(np.arange(0, n, 1), np.abs(Anexp),
           "ro", label="Coeff. An and Bn Exp")
plt.loglog(np.arange(0, n, 1), np.abs(Bnexp), "ro")
plt.title(r"Coeff. of fourier series of $e^x$ on a loglog scale")
plt.xlabel(r'$log(n)$')
plt.ylabel(r'$log(coeff)$')
plt.legend()
plt.grid()
plt.show()

plt.figure(5)
plt.semilogy(np.arange(0, n, 1), np.abs(
    Ancoscos), "ro", label="Coeff. An and Bn coscos")
plt.semilogy(np.arange(0, n, 1), np.abs(
    Bncoscos), "ro")
plt.title(r"Coeff. of fourier series of $cos(cos(x))$ on a semilogy scale")
plt.xlabel(r'$n$')
plt.ylabel(r'$log(coeff)$')
plt.legend()
plt.grid()
plt.show()

plt.figure(6)
plt.loglog(np.arange(0, n, 1), np.abs(Ancoscos),
           "ro", label="Coeff. An Coscos")
plt.loglog(np.arange(0, n, 1), np.abs(Bncoscos),
           "ro")
plt.title(r"Coeff. of fourier series of $cos(cos(x))$ on a loglog scale")
plt.xlabel(r'$log(n)$')
plt.ylabel(r'$log(coeff)$')
plt.grid()
plt.show()

x1 = np.linspace(0, 2*pi, 401)
b = exp(x1)
x1 = x1[:-1]           # drop last term to have a proper periodic integral
b = exp(x1)
A = np.zeros((400, 51))    # allocate space for A
A[:, 0] = 1             # col 1 is all ones
for k in range(1, 26):
    A[:, 2*k-1] = np.cos(k*x1)   # cos(kx) column
    A[:, 2*k] = np.sin(k*x1)  # sin(kx) column
    # endfor
c1 = sp.linalg.lstsq(A, b)[0]     # the ’[0]’ is to pull out the
# best fit vector. lstsq returns a list.
# print(c1)

temp = []
temp.append(Anexp[0])
for i in range(1, 26):
    temp.append(Anexp[i])
    temp.append(Bnexp[i])
plt.figure(7)

plt.semilogy(np.arange(0, 51, 1), np.abs(temp), "ro", label="true value")
plt.semilogy(np.arange(0, 51, 1), np.abs(c1), "og",
             label="Coeff. Least Sqr. Approx.")
plt.title(r"Coeff. of fourier series of $e^x$ on a semilogy scale")
plt.xlabel(r'$n$')
plt.ylabel(r'$log(coeff)$')
plt.legend()
plt.grid()
plt.show()

plt.figure(8)
plt.loglog(np.arange(0, 51, 1), np.abs(temp), "ro", label="true value")
plt.loglog(np.arange(0, 51, 1), np.abs(c1), "og",
           label="Coeff. Least Sqr. Approx.")
plt.title(r"Coefficients of fourier series of $e^x$ on a loglog scale")
plt.xlabel(r'$log(n)$')
plt.ylabel(r'$log(coeff)$')
plt.legend()
plt.grid()
plt.show()

temp1 = []
temp1.append(Ancoscos[0])
for i in range(1, 26):
    temp1.append(Ancoscos[i])
    temp1.append(Bncoscos[i])

b1 = coscos(x1)
A1 = np.zeros((400, 51))    # allocate space for A
A1[:, 0] = 1             # col 1 is all ones
for k in range(1, 26):
    A1[:, 2*k-1] = np.cos(k*x1)   # cos(kx) column
    A1[:, 2*k] = np.sin(k*x1)  # sin(kx) column
    # endfor
c2 = sp.linalg.lstsq(A1, b1)[0]

plt.figure(9)
plt.semilogy(np.arange(0, 51, 1), np.abs(temp1), "ro", label="true value")
plt.semilogy(np.arange(0, 51, 1), np.abs(c2), "og",
             label="Coeff. Least Sqr. Approx.")
plt.title(r"Coeff. of fourier series of $cos(cos(x))$ on a semilogy scale")
plt.xlabel(r'$n$')
plt.ylabel(r'$log(coeff)$')
plt.legend()
plt.grid()
plt.show()

plt.figure(10)
plt.loglog(np.arange(0, 51, 1), np.abs(temp1), "ro", label="true value")
plt.loglog(np.arange(0, 51, 1), np.abs(c2), "og",
           label="Coeff. Least Sqr. Approx.")
plt.title(r"Coeff. of fourier series of $cos(cos(x))$ on a loglog scale")
plt.xlabel(r'$log(n)$')
plt.ylabel(r'$log(coeff)$')
plt.legend()
plt.grid()
plt.show()

exp_dev = []
coscos_dev = []

for i in range(51):
    if i == 0:
        exp_dev.append(abs(c1[i]-temp[i]/2))
        coscos_dev.append(abs(c2[i]-temp1[i]/2))
    else:
        exp_dev.append(abs(c1[i]-temp[i]))
        coscos_dev.append(abs(c2[i]-temp1[i]))

print("Maximum deviation for exp(x): {}".format(max(exp_dev)))
print("Maximum deviation for coscos(x): {}".format(max(coscos_dev)))

exp_approx = np.dot(A, c1)
coscos_approx = np.dot(A1, c2)

plt.figure(11)
plt.semilogy(x1, list(map(abs, exp_approx)),
             "go", label="Function approximation")
plt.semilogy(x1, exp(x1), "ro", label="True Value")
plt.legend()
plt.grid()
plt.show()


plt.figure(12)
plt.plot(x1, list(map(abs, coscos_approx)),
         "go", label="Function approximation")
plt.plot(x1, coscos(x1), label="True Value")
plt.legend()
plt.grid()
plt.show()
