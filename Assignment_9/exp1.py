import numpy as np
import matplotlib.pyplot as plt

N = 100000


print("\n\Part I\n")


mu = 0
sigma = 1

print("\nQuestion 1\n")

r = sigma * np.random.randn(N) + mu

plt.figure(figsize=(8, 6))
plt.title("I.1")
plt.hist(r, bins=50, density=True)
x = np.linspace(np.min(r), np.max(r), N)
y = np.exp(-(x**2) / 2) / (2 * np.pi) ** 0.5
plt.plot(x, y)
plt.grid()

avg = 0
for value in r:
    avg += value
avg /= N

stdev = 0
for value in r:
    stdev += (value - avg) ** 2
stdev = (stdev / N) ** 0.5

print(f"Calculated Mean : {avg}")
print(f"Calculated Standard Deviation : {stdev}")
print(f"Mean using Built-in function : {r.mean()}")
print(f"Standard Deviation using Built-In function : {r.std()}\n")


print("\nQuestion 2\n")

one = two = three = 0
for value in r:
    if value > mu + 1 * sigma:
        one += 1
    if value > mu + 2 * sigma:
        two += 1
    if value > mu + 3 * sigma:
        three += 1

print(f"Percentage of Events > 1 sigma : {one * 100 / N}")
print(f"Percentage of Events > 2 sigma : {two * 100 / N}")
print(f"Percentage of Events > 3 sigma : {three * 100 / N}\n")


print("\nQuestion 3\n")

k = 1

avg = stdev = 0
for value in r:
    avg += value - k
    stdev += (value - k) ** 2
avg = avg / N + k
stdev = (stdev / N - (avg - k) ** 2) ** 0.5

print(f"Calculated Mean : {avg}")
print(f"Calculated Standard Deviation : {stdev}")
print(f"Mean using Built-in function : {r.mean()}")
print(f"Standard Deviation using Built-In function : {r.std()}\n")


print("\nQuestion 4\n")

r = np.random.rand(N)

plt.figure(figsize=(8, 6))
plt.title("I.4")
plt.hist(r, bins=200, density=True)
plt.grid()

print(f"Mean : {r.mean()}")
print(f"Standard Deviation : {r.std()}\n")
print(f"Expected Mean : {0.5}")
print(f"Expected Standard Deviation : {1 / 12 ** 0.5}\n")


print("\nQuestion 5\n")

r = np.random.rand(N) + np.random.rand(N)

plt.figure(figsize=(8, 6))
plt.title("I.5")
plt.hist(r, bins=200, density=True)
plt.grid()


print("\nQuestion 6\n")

r = 0
for i in range(5):
    r += np.random.rand(N)

plt.figure(figsize=(8, 6))
plt.title("I.6a")
plt.hist(r, bins=200, density=True)
plt.grid()

r = 0
for i in range(5):
    r += np.random.randn(N)

plt.figure(figsize=(8, 6))
plt.title("I.6b")
plt.hist(r, bins=200, density=True)
plt.grid()


print("\n\nPart II\n")


print("\nQuestion 1\n")

mu = 1
r = np.random.poisson(mu, N)
plt.figure(figsize=(8, 6))
plt.title("II.1")
plt.hist(r, bins=200, density=True)
plt.grid()

print(f"Mean : {r.mean()}")
print(f"Standard Deviation : {r.std()}\n")
print(f"Percentage of Events at mean : {sum(r == mu) * 100 / N}\n")


print("\nQuestion 2\n")

r = np.random.poisson(mu, N) + np.random.poisson(mu, N)

plt.figure(figsize=(8, 6))
plt.title("II.2")
plt.hist(r, bins=200, density=True)
plt.grid()


print("\nQuestion 3\n")

r = np.random.poisson(40, N)

plt.figure(figsize=(8, 6))
plt.title("II.3")
plt.hist(r, bins=200, density=True)
plt.grid()


print("\nQuestion 4\n")

N = 10
p = 0.2
r = np.random.binomial(N, p, 100000)

print(f"Mean : {r.mean()}")
print(f"Standard Deviation : {r.std()}")
print(f"Expected Mean : {N * p}")
print(f"Expected Standard Deviation : {(N * p * (1 - p)) ** 0.5}\n")


print("\nQuestion 5\n")

N = 10
p = (0.2, 0.8)
r = np.random.binomial(N, p[0], 100000) + np.random.binomial(N, p[1], 100000)

plt.figure(figsize=(8, 6))
plt.title("II.5")
plt.hist(r, bins=50, density=True)
plt.grid()


print("\nQuestion 6\n")

N = 200
p = 0.05
r = np.random.binomial(N, p, 100000)

plt.figure(figsize=(8, 6))
plt.title("II.6")
plt.hist(r, bins=25, density=True)
plt.grid()


plt.show()
