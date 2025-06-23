import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

signal, fs = sf.read("data/A1.wav")
N = len(signal)
T = N / fs
t = np.arange(0, T, 1 / fs)

ys = np.fft.fft(signal)[:N//2]
freq = np.linspace(0, fs, N)[:N//2]

energy = ys * np.conj(ys)
total_energy = np.sum(energy)
energy_perc = (energy / total_energy) * 100.0
ys[energy_perc < 1] = 0

non_zero_energy_freq = freq[np.abs(ys) > 0]
min_freq = np.min(non_zero_energy_freq)
max_freq = np.max(non_zero_energy_freq)

range_idx = np.where((freq >= min_freq) & (freq <= max_freq))

plt.stem(freq[range_idx], np.abs(ys)[range_idx])
plt.show()

# fs = 1000
#
# A = 1
# T = 0.5
# f1 = 20
# f2 = 40
# p = 0
#
# t = np.arange(0, T, 1 / fs)
# sin = A * np.sin(2 * np.pi * f1 * t + p)
# cos = A * np.cos(2 * np.pi * f2 * t + p)
#
# y = sin + cos
# N = len(y)
# ys = np.fft.fft(y)[:N//5]
# freq = np.linspace(0, fs, N)[:N//5]
#
#
#
#
# sin_result = np.dot(y, sin)
# cos_result = np.dot(y, cos)
#
# print(f"sin 20Hz : {np.round(sin_result, 2)}")
# print(f"cos 40Hz : {np.round(cos_result, 2)}")
#
# plt.stem(freq, np.abs(ys))
# plt.show()