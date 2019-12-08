#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import subprocess
import scipy.io as sio
import os

get_ipython().run_line_magic('run', "'./../split_step_fourier.ipynb'")
DEBUG = False
DEBUG_PLOT = False

# showing figures inline
get_ipython().run_line_magic('matplotlib', 'inline')
# plotting options 
figure_size = (20, 15)
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# parameters
f_symbol = 32e9  # symbol rate (Baud) (Symbols per second)
n_up = 10  # samples per symbol (>1 => oversampling)

r_rc = .33
syms_per_filt = 4  # symbols per filter (plus minus in both directions)
t_sample, ir = get_rc_ir(syms_per_filt, r_rc, f_symbol, n_up)

P_in = 19 # dBm

# modulation scheme and constellation points
M = 2
modulation = {'0': -1, '1': 1}
n_symbol = 30 # number of symbols

# Signalfolge generieren
send_bits = np.random.choice([symbol for symbol in modulation.keys()], size=n_symbol)

# Sendesignal generieren
send_ir = generate_signal(modulation, t_sample, 1/f_symbol, send_bits, ir, syms_per_filt, P_in)

# add zeros before and after signal (use samples per symbol as factor)
send = zeroing(send_ir, 10 * int(1/f_symbol/t_sample))


## Transmission parameters
z_length = 70  # [km]
nz = 10  # steps
dz = z_length / nz  # [km]

alpha = 0.2  # Dämpfung [dB/km]
D = 17  # [ps/nm/km]
beta2 = - (D * np.square(1550e-9)) / (2 * np.pi * 3e8) * 1e-3 # [s^2/km] propagation constant, lambda=1550nm is standard single-mode wavelength
gamma = 1.3 # [1/W/km]

# export inputs to matlab
sio.savemat('matlab/vars.mat', {'u0': send, 'dt': t_sample, 'dz': dz, 'nz': nz, 'alpha': alpha, 'betap': [0, 0, beta2], 'gamma': gamma})


# simulate python-transmission
output = splitstepfourier(send, t_sample, dz, nz, alpha, beta2, gamma)
# simulate matlab-transmission
print("Please execute matlab.sh in ./matlab")


# load matlab output
matlab = sio.loadmat('matlab/output.mat')
output_matlab = matlab['u'].flatten()


fig1, ax1 = plt.subplots(2, figsize=figure_size)

ax1[0].plot(np.arange(output.size)*t_sample, np.square(np.abs(output)), linestyle='--', linewidth=2.0, label='Python')
ax1[0].plot(np.arange(output_matlab.size)*t_sample, np.square(np.abs(output_matlab)), linestyle=':', linewidth=2.0, label='Matlab')
ax1[0].legend()

ax1[1].plot(np.fft.fftshift(np.square(abs(t_sample*np.fft.fft(output)/np.sqrt(2*np.pi)))), linestyle='--',linewidth=2.0,  label='Python')
ax1[1].plot(np.fft.fftshift(np.square(abs(t_sample*np.fft.fft(output_matlab)/np.sqrt(2*np.pi)))), linestyle=':', linewidth=2.0, label='Matlab')
ax1[1].legend()

print(f"Relativer Fehler: {calc_relerr(output, output_matlab)}")


tikzplotlib.save('../../../bachelorarbeit-ausarbeitung/figures/plots/compare_matlab.tex', figure=fig1, figureheight="\\figheight", figurewidth="\\figwidth")

