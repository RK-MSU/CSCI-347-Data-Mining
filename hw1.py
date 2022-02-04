import numpy as np
import math

data = np.ndarray((3, 7))
data[0] = np.array([0.3, 0.4, 1.8, 6.0, -0.5, 0.4, 1.1])
data[1] = np.array([23, 1, 4, 50, 34, 19, 11])
data[2] = np.array([5.6, 5.2, 5.2, 5.1, 5.7, 5.4, 5.5])


data = np.ndarray((7, 3))
data[0] = np.array([0.3, 23, 5.6])
data[1] = np.array([0.4, 1, 5.2])
data[2] = np.array([1.8, 4, 5.2])
data[3] = np.array([6.0, 50, 5.1])
data[4] = np.array([-0.5, 34, 5.7])
data[5] = np.array([0.4, 19, 5.4])
data[6] = np.array([1.1, 11, 5.5])

# print(data)

X1 = data[:,0]
X2 = data[:,1]
X3 = data[:,2]

print('2.1) {:.2f}'.format(X3.mean()))

def covar(a, b):
    answer = 0
    for i in range(data.shape[0]):
        answer = answer + (a[i] - a.mean()) * (b[i] - b.mean())
    return answer / (data.shape[0] - 1)

print('2.2) {:.2f}'.format(covar(X1, X3)))

print("2.3) ({:.2f} {:.2f} {:.2f})".format(X1.mean(), X2.mean(), X3.mean()))

print('2.4) {:.2f}'.format(np.var(X2, ddof=1)))

print('2.5)\n  {:.2f} {:.2f} {:.2f}\n  {:.2f} {:.2f} {:.2f}\n  {:.2f} {:.2f} {:.2f}'.format(
    np.var(X1, ddof=1), covar(X1, X2), covar(X1, X3),
    covar(X2, X1), covar(X2, X2), covar(X2, X3),
    covar(X3, X1), covar(X3, X2), covar(X3, X3)))

print('2.6) {:.2f}'.format(covar(X1, X3) / (math.sqrt(covar(X1, X1)) * math.sqrt(covar(X3, X3)))))

answer = 0
for i in range(data.shape[1]):
    answer += np.var(data[:,i], ddof=1)
print('2.7) ', answer)
