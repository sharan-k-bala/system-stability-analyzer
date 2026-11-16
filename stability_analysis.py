import numpy as np
import matplotlib.pyplot as plt

#now we'll define the matrix
A = np.array([[0.8, 0.1],[0.2, 0.7]])

#now we'll define the initial state
X = np.array([1.0, 2.0])

#calculating the eigenvalues and eigen vectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("System matrix:")
print(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors")
print(eigenvectors)

#now we'll determine the system stability in this step
if np.all(np.abs(eigenvalues) < 1):
    print("\nThe system is Asymtotically stable")
elif np.any(np.abs(eigenvalues) > 1):
    print("\nThe system is unstable")
else:
    print("\nThe system is Marginally stable")

#simulating the system
time_steps = 20

states = []

x = X.copy()


for t in range(time_steps):
    states.append(x.copy())
    x = A @ x

states = np.array(states)

#plot the system behaviour
plt.plot(states[:, 0], label="State 1")
plt.plot(states[:, 1], label="State 2")

plt.xlabel("Time steps")
plt.ylabel("State values")
plt.title("System stability analysis")
plt.legend()
plt.grid()

plt.show()