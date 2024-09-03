#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  12 23:53:25 2024
@author: satoshihoshika
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1
import sys

# 2次元グリーン関数の定義 / Definition of 2D Green's function
def green2d(xr, yr, xs, ys, k):
    Xs = xs * np.ones(xr.shape)
    Ys = ys * np.ones(yr.shape)
    r = np.sqrt((xr - Xs) ** 2 + (yr - Ys) ** 2)
    G = (1.j / 4.) * hankel1(0., k * r)
    return G

# フーリエ変換行列の生成 / Fourier matrix generation
def fourierMatMN(M, N):
    F = np.zeros((M, N), dtype=complex)  # Fourier matrix
    Finv = np.zeros((N, M), dtype=complex)  # Inverse Fourier matrix
    wavenum = np.linspace(0, N - 1, M)

    for m in range(M):
        for n in range(N):
            F[m, n] = np.exp(2j * np.pi * wavenum[m] * n / N)
            Finv[n, m] = np.exp(-2j * np.pi * wavenum[m] * n / N) / N

    return F, Finv

##########################################################################
# パラメータ設定 / Parameter settings
##########################################################################
f = 3400  # frequency [Hz]
c = 340   # speed of sound [m/s]
k = 2 * np.pi * f / c  # wave number
# condition = 'specular'  # 'specular' or 'bragg'
condition = 'bragg'  # 'specular' or 'bragg'

# サウンドフィールドのサイズ [m] / Size of the sound field [m]
lenX = 5.0
lenY = 0.1

# サウンドフィールドの間隔 [m] / Interval of the sound field [m]
dx = 0.05
dy = 0.02

# サンプル数 / Number of samples
Nx = int(np.round(lenX / dx))
Ny = int(np.round(lenY / dy))

fs_sp = c / dx  # 空間サンプリング周波数 / Spatial sampling frequency

# 受音位置 / Receiver positions
xr = np.arange(0, Nx) * dx
yr = np.arange(0, Ny) * dy

# 2次元グリッド / 2D grid
Xr = np.tile(xr, (Ny, 1)).T
Yr = np.tile(yr, (Nx, 1))

M = 400  # 音源の数 / Number of sound sources
N = len(xr)  # 受音機の数 / Number of receivers

# 音源位置 / Sound source positions
xs = np.random.uniform(-lenX, lenX, M)
ys = np.random.uniform(0.1, 0.5, M).reshape(M, 1)

# 伝達関数の初期化 / Initialization of transfer functions
H_i = np.zeros((N, M), dtype='complex')  # 入射波 / Incident wave
H_r = np.zeros((N, M), dtype='complex')  # 反射波 / Reflected wave
H_r2 = np.zeros((N, M), dtype='complex') # 反射波（別の条件）/ Reflected wave (alternative condition)

Bins = 100  # 波数スペクトルビン数 / Number of wavenumber spectrum bins
kx = 2 * np.pi * np.linspace(-Nx / 2, Nx / 2 - 1, Nx) / (Nx * dx)
ky = np.emath.sqrt(k ** 2 - kx ** 2)
kx = np.fft.fftshift(kx)
ky = np.fft.fftshift(ky)

maxBin = int(Nx * f / fs_sp)

##########################################################################
# Pi, Pr, Crの計算 / Calculation of Pi, Pr, Cr
##########################################################################
F, Finv = fourierMatMN(Bins, Nx)

# 入射波の計算 / Calculation of the incident wave
for m in range(M):
    H_i[:, m] = green2d(Xr[:, 0], Yr[:, 0], xs[m], ys[m], k)

# 鏡像ソースの位置 / Position of the mirror image source
ys_r = -ys
d = 4 * c / f / 5

# 反射波の計算 / Calculation of the reflected wave
for i in range(M):
    H_r[:, i] = green2d(Xr[:, 0], Yr[:, 0], xs[i], ys_r[i, 0], k)
    H_r2[:, i] = green2d(Xr[:, 0], Yr[:, 0], xs[i], ys_r[i, 0] - 2 * d, k)

# Piの計算 / Calculation of Pi
Pi = np.zeros((Bins, M), dtype='complex')
pi = H_i
Pi_bound = np.zeros((maxBin * 2, M), dtype='complex')

for i in range(len(xs)):
    Pi[:, i] = F @ pi[:, i]
    Pi_bound[0:maxBin, i] = Pi[0:maxBin, i]
    Pi_bound[-maxBin:, i] = Pi[-maxBin:, i]

# Prの計算 / Calculation of Pr
if condition == 'specular':
    pr = 0.5 * H_r  # 鏡面反射 / Specular reflection
elif condition == 'bragg':
    pr = 0.5 * H_r + 0.5 * H_r2  # ブラッグ反射 / Bragg reflection
else:
    print('condition error / please set specular or bragg for condition')
    sys.exit()

Pr = np.zeros((Bins, M), dtype='complex')
Pr_bound = np.zeros((maxBin * 2, len(xs)), dtype='complex')

for i in range(M):
    Pr[:, i] = F @ pr[:, i]
    Pr_bound[0:maxBin, i] = Pr[0:maxBin, i]
    Pr_bound[-(maxBin):, i] = Pr[-(maxBin):, i]

rank_Pi = np.linalg.matrix_rank(Pi)
rank_Pr = np.linalg.matrix_rank(Pr)

Cr = Pr @ np.linalg.pinv(Pi) * Bins / Nx
Cr_bound = Pr_bound @ np.linalg.pinv(Pi_bound)

##########################################################################
# 理論解の計算 / Analytical solution calculation
##########################################################################
theta = np.linspace(-90, 90, 1000)
x = np.linspace(-2.5, 2.5, 1000)
y = np.zeros(x.shape)

r = np.vstack((x, y))
lam = c / f
k2 = 2 * np.pi / lam
kx2 = k2 * np.cos(theta / 180 * np.pi)
ky2 = np.sqrt(k2 ** 2 - kx2 ** 2)
diff = 2 * d * np.cos(theta / 180 * np.pi)  # 行路差 / Path difference

pi_theo = np.zeros((len(x), len(theta)), dtype=complex)
pr_theo = np.zeros((len(x), len(theta)), dtype=complex)

for i in range(len(theta)):
    pi_theo[:, i] = 1.0 * np.exp(-1j * (kx2[i] * x + ky2[i] * y))

    if condition == 'specular':
        pr_theo[:, i] = 0.5 * np.exp(-1j * (kx2[-i] * x + ky2[-i] * y))  # 鏡面反射 / Specular reflection
    elif condition == 'bragg':
        pr_theo[:, i] = 0.5 * np.exp(-1j * (kx2[-i] * x + ky2[-i] * y)) + \
                        0.5 * np.exp(-1j * (kx2[-i] * x + ky2[-i] * y)) * np.exp(1j * k2 * diff[i])  # ブラッグ反射 / Bragg reflection
    else:
        print('condition error')

Cr_theo = pr_theo[0, :] / pi_theo[0, :]
Cr_calc = np.diag(Cr)

##########################################################################
# コサイン類似度の計算 / Cosine similarity calculation
##########################################################################
kx_calc = 2 * np.pi * np.linspace(-Bins / 2, Bins / 2 - 1, Bins) / (Bins * dx)
theta_calc = np.arccos(kx_calc / k)
theta_calc = np.nan_to_num(theta_calc)
theta_calc = np.fft.fftshift(theta_calc) / np.pi * 180 - 90

pi_theo_sim = np.zeros((len(x), len(theta_calc)), dtype=complex)
pr_theo_sim = np.zeros((len(x), len(theta_calc)), dtype=complex)

diff = 2 * d * np.cos(theta_calc / 180 * np.pi)  # 行路差 / Path difference

for i in range(len(theta_calc)):
    pi_theo_sim[:, i] = 1.0 * np.exp(-1j * (kx[i] * x + ky[i] * y))
    pr_theo_sim[:, i] = 0.5 * np.exp(-1j * (kx[-i] * x + ky[-i] * y)) + \
                        0.5 * np.exp(-1j * (kx[-i] * x + ky[-i] * y)) * np.exp(1j * k * diff[i])  # ブラッグ反射 / Bragg reflection

Cr_theo_sim = np.abs(pr_theo_sim[0, :] / pi_theo_sim[0, :])
Cr_calc = np.abs(Cr_calc)

# コサイン類似度の計算 / Cosine similarity calculation
cossim = np.dot(Cr_theo_sim, Cr_calc) / (np.linalg.norm(Cr_theo_sim, ord=2) * np.linalg.norm(Cr_calc, ord=2))
print('Cosine similarity: ', cossim)

##########################################################################
# 二乗誤差の計算 / Square error calculation
##########################################################################
error = np.sum((Cr_theo_sim - Cr_calc) ** 2) / len(Cr_theo_sim)

##########################################################################
# 図の描画 / Plotting figures
##########################################################################
plt.rcParams['font.family'] = 'Times New Roman'  # フォントの設定 / Font setting
plt.figure()
tmp = np.abs(np.fft.fftshift(Cr_bound))
plt.imshow(tmp, vmin=0, vmax=1)
plt.title('Cr_bound')
plt.colorbar()

plt.figure()
plt.imshow(np.abs(np.fft.fftshift(Cr)), vmin=0, vmax=1)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(Pi))
plt.title('Pi')
plt.colorbar()

plt.figure()
plt.imshow(np.abs(Pr))
plt.title('Pr')
plt.colorbar()

plt.figure()
plt.imshow(np.abs(Pr / Pi))
plt.title('Pr/Pi')
plt.colorbar()

##########################################################################
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.size"] = 15  # 全体のフォントサイズ / Overall font size
plt.rcParams['xtick.labelsize'] = 14  # x軸のラベルサイズ / x-axis label size
plt.rcParams['ytick.labelsize'] = 14  # y軸のラベルサイズ / y-axis label size
plt.rcParams['xtick.direction'] = 'in'  # x軸の方向 / x-axis direction
plt.rcParams['ytick.direction'] = 'in'  # y軸の方向 / y-axis direction
plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅 / Axis linewidth
plt.rcParams['axes.grid'] = True  # グリッドの表示 / Enable grid

# 結果の描画 / Plot results
plt.figure()
plt.scatter(theta_calc, np.abs(Cr_calc), marker='x', c='c')
plt.plot(theta, np.abs(Cr_theo), c='black')
plt.xlim((-90, 90))
plt.ylim((0, 1))
plt.xticks(np.arange(-90, 91, 15))
plt.xlabel(r"$\theta$ [deg]")
plt.ylabel(r"diag($C\rm{r}$)")
plt.legend(['Estimated value', 'Analytical solution'], loc='lower left', bbox_to_anchor=(.5, 1.), ncol=1)




