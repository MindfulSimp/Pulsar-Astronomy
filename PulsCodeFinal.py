import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

N = 512
M = 60
P = 1000
samples = N*M*P

north, south = np.zeros(samples), np.zeros(samples)
with open("demo.txt") as f:
    for i, line in enumerate(f):
        if i==samples: break
        n,s = map(float, line.rstrip().split(' '))
        north[i] = n
        south[i] = s
# print(north)
# print(south)

sampling_interval = 30.30*10**(-9)
x_axis = np.arange(0, samples, 1)*sampling_interval
plt.plot(x_axis, north)
plt.show()

cur=0
res = np.zeros(N, dtype='complex128')
for i in range(M):
    X = fft(north[cur:cur+N])**2
    cur+=N
    res += X
res/=M
freq = np.arange(N)/sampling_interval
plt.plot(freq[-N//2:], np.abs(res)[-N//2:])
plt.show()

spectra = np.zeros((P, N//2))
cur = 0
for j in range(P):
    res = np.zeros(N, dtype='complex128')
    for i in range(M):
        X = fft(north[cur:cur+N])**2
        cur+=N
        res += X
    res/=M
    spectra[j]=np.abs(res[-N//2:])

fig, (ax0, ax1) = plt.subplots(2, 1)
spectra = np.transpose(spectra)
shift, cs = 280, 0
delta = shift/256
shp = spectra.shape

shifted = np.full((shp[0], shp[1]+shift), 50000)
for i, r in enumerate(spectra):
    shifted[i][int(cs):int(cs)+shp[1]] = r
    cs+=delta

colorplot_shifted = ax1.pcolor(shifted, cmap='magma')
colorplot = ax0.pcolor(spectra, cmap='magma')
ax0.set_title('output')
fig.colorbar(colorplot_shifted, ax=ax1)
fig.colorbar(colorplot, ax=ax0)
fig.tight_layout()
plt.show()