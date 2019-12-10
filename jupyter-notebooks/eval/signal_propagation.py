#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
import tikzplotlib

get_ipython().run_line_magic('run', "'./../split_step_fourier.ipynb'")
DEBUG = False

# showing figures inline
get_ipython().run_line_magic('matplotlib', 'inline')
# plotting options 
figure_size = (10, 10*np.sqrt(2))
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
t_sample_rc, rc = get_rc_ir(syms_per_filt, r_rc, f_symbol, n_up)

power = 5

# modulation scheme and constellation points
M = 2
modulation = {'0': -1, '1': 1}
n_symbol = 30 # number of symbols


# Signalfolge generieren
send_bits = np.random.choice([symbol for symbol in modulation.keys()], size=n_symbol)


# Sendesignal generieren
send_rc = generate_signal(modulation, t_sample_rc, 1/f_symbol, send_bits, rc, syms_per_filt, power)

# add zeros before and after signal (use samples per symbol as factor)
send_new = zeroing(send_rc, 5 * int(1/f_symbol/t_sample_rc))


## Transmission
z_length = 70  # [km]
nz = 10  # steps
dz = z_length / nz  # [km]

alpha = 0.2  # Dämpfung [dB/km]
D = 17  # [ps/nm/km]
beta2 = - (D * np.square(1550e-9)) / (2 * np.pi * 3e8) * 1e-3 # [s^2/km] propagation constant, lambda=1550nm is standard single-mode wavelength
gamma = 1.3 # [1/W/km]

output = splitstepfourier(send_new, t_sample_rc, dz, nz, alpha, beta2, gamma, True)


fig1, ax1 = plt.subplots(len(output)+1, figsize=(15,30), sharex=True)

colors = cycle(list(mcolors.TABLEAU_COLORS))
counter = 0

x_vals = np.arange(send_new.size)*t_sample_rc
xmin = np.amin(x_vals)
xmax = np.amax(x_vals)

ax1[counter].plot(x_vals, np.square(np.abs(send_new)), label='0', color=next(colors))
ax1[counter].set_xlim(xmin, xmax)
ax1[counter].set_ylim(bottom=0)
ax1[counter].set_title("Input (0 km)")
ax1[counter].set_ylabel("$|s|^2$")
counter += 1

for key, val in output.items():
    ax1[counter].plot(x_vals, np.square(np.abs(val)), label=key, color=next(colors))
    ax1[counter].set_ylim(bottom=0)
    ax1[counter].set_title(f"after {key} steps ({float(key)*dz} km)")
    ax1[counter].set_ylabel("$|s|^2$")
    counter += 1

ax1[counter-1].set_title(f"Output ({z_length} km)")
ax1[counter-1].set_xlabel("t [s]")


## Transmission
z_length = 70  # [km]
nz = 10  # steps
dz = z_length / nz  # [km]

alpha = 0  # Dämpfung [dB/km]
D = 17  # [ps/nm/km]
beta2 = - (D * np.square(1550e-9)) / (2 * np.pi * 3e8) * 1e-3 # [s^2/km] propagation constant, lambda=1550nm is standard single-mode wavelength
gamma = 1.3 # [1/W/km]

output2 = splitstepfourier(send_new, t_sample_rc, dz, nz, alpha, beta2, gamma, True)


fig2, ax2 = plt.subplots(len(output)+1, figsize=(15,30), sharex=True)

colors = cycle(list(mcolors.TABLEAU_COLORS))
counter = 0

all_vals = np.square(np.abs(np.asarray([val for val in output2.values()]).flatten()))
ymin = np.amin(all_vals)
ymax = np.amax(all_vals)*1.1

x_vals = np.arange(send_new.size)*t_sample_rc
xmin = np.amin(x_vals)
xmax = np.amax(x_vals)

ax2[counter].plot(x_vals, np.square(np.abs(send_new)), label='0', color=next(colors))
ax2[counter].set_xlim(xmin, xmax)
ax2[counter].set_ylim(ymin, ymax)
ax2[counter].set_title(f"Input")
ax2[counter].set_ylabel("$|s|^2$")
counter += 1

for key, val in output2.items():
    ax2[counter].plot(x_vals, np.square(np.abs(val)), label=key, color=next(colors))
    ax2[counter].set_xlim(xmin, xmax)
    ax2[counter].set_ylim(ymin, ymax)
    ax2[counter].set_title(f"after {key} steps ({float(key)*dz} km)")
    ax2[counter].set_ylabel("$|s|^2$")
    counter += 1
    
ax2[counter-1].set_title(f"Output ({z_length} km)")
ax2[counter-1].set_xlabel("t [s]")


output_fname = "fiber_propagation"
output_path = "../../../bachelorarbeit-ausarbeitung/figures/plots/"

tikzplotlib.save(f'{output_path}{output_fname}.tex', figure=fig1, figureheight="\\figheight", figurewidth="\\figwidth")
tikzplotlib.save(f'{output_path}{output_fname}_noalpha.tex', figure=fig2, figureheight="\\figheight", figurewidth="\\figwidth")

fig1.savefig(f"{output_path}{output_fname}.pdf", bbox_inches='tight')
fig2.savefig(f"{output_path}{output_fname}_noalpha.pdf", bbox_inches='tight')

