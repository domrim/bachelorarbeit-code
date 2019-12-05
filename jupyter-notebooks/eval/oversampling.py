#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

get_ipython().run_line_magic('run', "'./../split_step_fourier.ipynb'")
DEBUG = False
DEBUG_PLOT = True

# showing figures inline
get_ipython().run_line_magic('matplotlib', 'inline')
# plotting options 
font = {'size': 12}
figure_size = (25, 15)
plt.rc('font', **font)
plt.rc('text', usetex=True)


# parameters
f_symbol = 32e9  # symbol rate (Baud) (Symbols per second)

r_rc = .33
syms_per_filt = 4  # symbols per filter (plus minus in both directions)

P_in = 19 # dBm

# modulation scheme and constellation points
M = 2
modulation = {'0': -1, '1': 1}
n_symbol = 30 # number of symbols

# Signalfolge generieren
send_bits = np.random.choice([symbol for symbol in modulation.keys()], size=n_symbol)


## Transmission parameters
z_length = 70  # [km]
nz = 10  # steps
dz = z_length / nz

alpha = 0.2  # Dämpfung [dB/km]
D = 17  # [ps/nm/km]
beta2 = - (D * np.square(1550e-9)) / (2 * np.pi * 3e8) * 1e-3 # [s^2/km] propagation constant, lambda=1550nm is standard single-mode wavelength
gamma = 1.3 # [1/W/km]


## Simulation
n_up_max = 16  # samples per symbol (>1 => oversampling)

outputs = {}
sampletimes = {}

for n_up in range(1, n_up_max+1):
    # Pulse IR
    t_sample, ir  = get_rc_ir(syms_per_filt, r_rc, f_symbol, n_up)
    # Sendesignal generieren
    send_ir = generate_signal(modulation, t_sample, 1/f_symbol, send_bits, ir, syms_per_filt, P_in)
    # add zeros before and after signal (use samples per symbol as factor)
    send = zeroing(send_ir, 10 * int(1/f_symbol/t_sample))
    # transmission
    output = splitstepfourier(send, t_sample, dz, nz, alpha, beta2, gamma)

    outputs[f"{n_up}"] = output
    sampletimes[f"{n_up}"] = t_sample
    


fig1, ax1 = plt.subplots(1, figsize=figure_size)

for key, item in outputs.items():
    if key in ['1', '2', '3', '4', '8', '16']:
        x_vals = np.arange(item.size)*sampletimes[key]
        ax1.plot(x_vals, np.square(np.abs(item)), label=key)
    
ax1.legend()


fig2, ax2 = plt.subplots(1, figsize=figure_size)

for key, item in outputs.items():
    if key in ['1', '2', '3', '4', '8', '16']:
        x_vals = np.arange(item.size)*sampletimes[key]
        ax2.plot(x_vals, np.square(np.abs(item)), label=key)
    
ax2.legend()
ax2.set_xlim(0.5e-9, 0.75e-9)


tikzplotlib.save('../../../bachelorarbeit-ausarbeitung/figures/plots/oversampling_full.tex', figure=fig1)
tikzplotlib.save('../../../bachelorarbeit-ausarbeitung/figures/plots/oversampling_zoom.tex', figure=fig2)
