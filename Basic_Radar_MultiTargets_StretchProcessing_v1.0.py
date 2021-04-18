# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:55:40 2020

@author: minhthuy.pham
"""

import numpy as np
import scipy.linalg as la
from numpy.linalg import*
import matplotlib.pyplot as plt
from scipy.signal import*
from numpy.fft import*

#Mixer
def Mixer(f0, f1, fa, td, trec, dr, pulse):
    #create reference LFM
    t =  np.linspace(0,td +trec, int(np.ceil((td +trec)*fa)))
    x_ref = chirp(t, f0, td, f0 + f1)
    t_dr = dr*2/c
    nmax_zeros = int(np.ceil(t_dr*fa))
    sig_size = np.ma.size(x_ref)
    nr_out = np.linspace(0, 1/fa*(sig_size-1), sig_size)
    pulse_size = np.ma.size(pulse)
    x_n = pulse + 1*np.random.randn(pulse_size)
    x_r = np.zeros(sig_size)
    x_r[(nmax_zeros):( nmax_zeros+pulse_size)] = pulse
    out = x_r*x_ref
    return out
#LPF 
def LPFilter(f1, fa, td, trec, mixer_out, order):
    #create low pass filter
    t =  np.linspace(0,td +trec, int(np.ceil((td +trec)*fa)))
    wn = (2*f1*trec/td)
    b, a = butter(order, wn, 'low', analog = True) #True can be in quotation marks
    f_tf = TransferFunction(b,a) 
    out = lsim(f_tf.to_ss(), mixer_out, t)
    return out

def CalculateLPFilter(c, td, trec, f1, f0, fa, order, d, dr):
    tret =  np.linspace(0,td, int(np.ceil(td*fa)))
    x = chirp(tret, f0, td, f0 + f1)
    mixer_out = Mixer(f0, f1, fa, td, trec, dr, x)
    out = LPFilter(f1, fa, td, trec, mixer_out, order)
    return out

#Param
def CalFFTParam(fa, td, trec):
    fmax = f1*trec/td
    f0 = 1/td
    fs = fa/10**np.floor(np.log10(fa/(2*fmax)))
    N = int(2**np.ceil(np.log2(fs/f0)))
    return N, fs
def SamplingLPF(fa, fs, N, out_LPF):
    r_f = int(fa/fs)
    out_sampling = np.zeros(N)
    for it in range(0, N):
        out_sampling[it] = out_LPF[r_f*it]
    return out_sampling
def IFFTOut(N, fs, out_sampling):
    out_ifft = fftshift((fft(out_sampling)))
    f0 = fs/N
    frq_bins = np.linspace(-(N/2)*f0,(N/2-1)*f0, N)
    return out_ifft, frq_bins

#Input
c = 3*np.power(10,8)
td = 10*10**-6
f1 = 500*10**6
f0 = 1*10**9
fa = 10*f0
d = 100
order = 3
n_obj = 3
dr = np.zeros(n_obj)
dr[0] = 3
dr[1] = 10
dr[2] = 50
trec = 2*d/c
t_r = np.linspace(0,td +trec, int(np.ceil((td +trec)*fa)))
sig_size = np.size(t_r)
x_r = np.zeros((n_obj, sig_size))
trec = 2*d/c
for it in range(0,n_obj):
    temp = CalculateLPFilter(c, td, trec, f1, f0, fa, order, d, dr[it])
    x_r[it,:] = temp[1]
#output of the LPF
sig_com = np.sum(x_r, axis = 0)

#Calculate the params for the FFT
N, fs = CalFFTParam(fa, td, trec)
#Sampling the signal from LPF
sig_LPF = SamplingLPF(fa, fs, N, sig_com)
#output of the FFT
sig_FFT, frq_bins = IFFTOut(N, fs, sig_LPF)

#Output & Freq
plt.figure()
plt.subplot(211)
plt.plot(frq_bins[int(N/2):N]*10**-6, 20*np.log10(np.abs(sig_FFT[int(N/2):N])))
plt.grid(True)
plt.xlabel('Frequency(MHz)')
plt.ylabel('Magnitude(dB)')
plt.show()

# Output & distance
plt.subplot(212)
dist = frq_bins*c/2*td/f1
plt.plot(dist[int(N/2):N], 20*np.log10(np.abs(sig_FFT[int(N/2):N])))
plt.grid(True)
plt.xlabel('Distance (m)')
plt.ylabel('Output')
plt.show()




