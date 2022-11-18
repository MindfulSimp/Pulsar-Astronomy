from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from scipy.stats import norm
import os

os.system("cls")
N = 512
M = 60
P = 2000
samples = N*M*P
north, south = np.zeros(samples), np.zeros(samples)
with open("ch00_B0833-45_20150612_191438_010_4.txt") as f:
    print("loading file..")
    for i, line in enumerate(f):
        if i==samples: break
        n,s = map(float, line.rstrip().split(' '))
        north[i] = n
        south[i] = s
# print(north)
# print(south)

sampling_interval = 30.30*10**(-9)
# x_axis = np.arange(0, samples, 1)*sampling_interval
# plt.plot(x_axis, north)
# plt.show()

# mx, mn = np.max(north), np.min(north)
# _mean, _variance = np.mean(north), np.var(north)
# print(_mean, _variance, mx, mn)
# x_axis_gaussan = np.arange(mn-1, mx+1, 0.1)
# plt.plot(x_axis_gaussan, norm.pdf(x_axis_gaussan,_mean,_variance**(1/2)))
#    res = np.zeros(N, dtype='complex128')

# cur=0
# res = np.zeros(N, dtype='complex128')
# for i in range(M):
#     # X = fft(north[i*(N):(i+1)*(N)])
#     X = fft(north[cur:cur+N])**2
#     cur+=N
#     # print(res[0], X[0])
#     res += X
# res/=M
# freq = np.arange(N)/sampling_interval
# plt.plot(freq[-N//2:], np.abs(res)[-N//2:])
# plt.show()

spectrum = np.zeros((P, N//2))
cur = 0
for j in range(P):
    os.system('cls')
    print(f"calculating fft...\n{str(((j+1)*100)//P) + '% done'}")
    res = np.zeros(N, dtype='complex128')
    for i in range(M):
        # X = fft(north[i*(N):(i+1)*(N)])
        X = fft(north[cur:cur+N])**2
        cur+=N
        # print(res[0], X[0])
        res += X
    res/=M
    # freq = np.arange(N)/sampling_interval
    spectrum[j]=np.abs(res[-N//2:])
    # plt.plot(freq[-N//2:], np.abs(res)[-N//2:]**2)
    # plt.show()
fig, (ax0, ax1) = plt.subplots(2, 1)
spectrum = np.transpose(spectrum)

shift, cs = 280, 0
delta = shift/256
shp = spectrum.shape
shifted = np.full((shp[0], shp[1]+shift), 50000)
for i, r in enumerate(spectrum):
    shifted[i][int(cs):int(cs)+shp[1]] = r
    cs+=delta

colorplot_shifted = ax1.pcolor(shifted, cmap='magma')
colorplot = ax0.pcolor(spectrum, cmap='magma')

ax0.set_title('output')
fig.colorbar(colorplot_shifted, ax=ax1)
fig.colorbar(colorplot, ax=ax0)
fig.tight_layout()
plt.show()
