import numpy as np
import matplotlib.pyplot as plt

# this script was blindly ported from Matlab to Python...

def custom_gamma(a, b, t, beta):
    x = np.random.gamma(a * t**beta - a * (t - 1)**beta, 1 / b)
    return x


mT = 0.4
beta = 1.5
st = 0.2 * mT
T = 70

N = 1000

varT = st * st
b = mT / varT
a = (mT * mT / varT) / (np.power(T, beta))

D = np.zeros((N, T + 1))
to = np.zeros((1, N))

for i in range(N):
    to[0][i] = 1
    D[i, 0] = 0

    for t in range(1, T):
        # dD = gamrnd(a*t^beta-a*(t-1)^beta,1/b);
        dD = custom_gamma(a, b, t, beta)
        D[i, t+1] = D[i, t] + dD
        D[i, t+1] = min(.9999999, float(D[i, t+1]))

    plt.plot(D[i, :])

D_mean = np.mean(D, axis=0)
plt.plot(D_mean, linewidth=4, color="red")
plt.show()

bin_size = 2
n_bins = int(100 / bin_size)
Dlabel = D.copy()

for k in range(N):
    for l in range(int(to[0][k]), T + 1):
        for j in range(n_bins + 1):
            if (D[k, l] < (j * bin_size / 100)) and (D[k, l] >= (j - 1) * bin_size / 100):
                Dlabel[k, l] = j

trans_counts = np.zeros((n_bins, n_bins, T))

for i in range(N):
    for t in range(int(to[0][i]), T + 1):
        trans_counts[int(Dlabel[i, t - 1]), int(Dlabel[i, t]), t - 1] =\
            1 + trans_counts[int(Dlabel[i, t - 1]), int(Dlabel[i, t]), t - 1]

counts = np.sum(trans_counts, axis=1, keepdims=True)

p = trans_counts.copy()

for i in range(n_bins):
    for t in range(1, T+1):
        p[i, :, t - 1] = trans_counts[i, :, t - 1] / counts[i, 0, t - 1]

print("bla")