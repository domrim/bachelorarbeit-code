#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

get_ipython().run_line_magic('run', "'./../split_step_fourier.ipynb'")
DEBUG = False
DEBUG_PLOT = False

# showing figures inline
get_ipython().run_line_magic('matplotlib', 'inline')
# plotting options 
figure_size = (25, 15)
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

P_in = 5 # dBm

# modulation scheme and constellation points
M = 2
modulation = {'0': -1, '1': 1}
n_symbol = 30 # number of symbols


# Signalfolge generieren
send_bits = np.random.choice([symbol for symbol in modulation.keys()], size=n_symbol)


# Sendesignal generieren
send_ir = generate_signal(modulation, t_sample, 1/f_symbol, send_bits, ir, syms_per_filt)

# add zeros before and after signal (use samples per symbol as factor)
send = zeroing(send_ir, 80 * int(1/f_symbol/t_sample))


## Transmission parameters
full_distance = 800

z_length = 70  # [km]
nz = 10  # steps
dz = z_length / nz  # [km]

alpha = 0.2  # Dämpfung [dB/km]
D = 17  # [ps/nm/km]
beta2 = - (D * np.square(1550e-9)) / (2 * np.pi * 3e8) * 1e-3 # [s^2/km] propagation constant, lambda=1550nm is standard single-mode wavelength
gamma = 1.3 # [1/W/km]


# init arrays for plotting with values at 0 input
power_full_distance = [np.power(10, (P_in-30)/10)]
xvals_full_distance = [0.0]

# put an amplifier every z_length km in the fiber
next_input = amplifier(send, P_in, t_sample, 1/f_symbol)

if DEBUG:
    print(f"Complete Fiber: {full_distance} km")
    print(f"Subsegment length: {z_length}")
    print(f"Subsegments: {full_distance/z_length}")
    print(f"Complete Subsegments: {int(full_distance/z_length)}")
    print(f"Rest segment length: {full_distance%z_length}")

for segment in range(int(full_distance / z_length)):
    segment_output = splitstepfourier(next_input, t_sample, dz, nz, alpha, beta2, gamma, True)
    segment_power = [np.real(calc_power(val, t_sample, 1/f_symbol)) for val in segment_output.values()]
    segment_xvals = [float(key)*dz + segment * z_length for key in segment_output.keys()]
    
    if DEBUG:
        print(f"Segment {segment+1} power values: {segment_power}")
    if DEBUG_PLOT:
        plt.plot(np.real(segment_output[f"{str(nz)}"]), label="Real")
        plt.plot(np.imag(segment_output[f"{str(nz)}"]), label="Imag")
        plt.legend()
        plt.show()

    next_input = amplifier(segment_output[f"{str(nz)}"], P_in, t_sample, 1/f_symbol)    
    
    power_full_distance += segment_power
    xvals_full_distance += segment_xvals
    
    if segment is not int(full_distance/z_length)-1:
        power_full_distance += [np.real(calc_power(next_input, t_sample, 1/f_symbol))]
        xvals_full_distance += [(segment+1.0)*z_length]

if full_distance % z_length != 0:
    power_full_distance += [np.real(calc_power(next_input, t_sample, 1/f_symbol))]
    xvals_full_distance += [(segment+1.0)*z_length]
    
    rest_z_length = full_distance % z_length
    rest_dz = rest_z_length / nz
    rest_output = splitstepfourier(next_input, t_sample, rest_dz, nz, alpha, beta2, gamma, True)
    rest_power = [np.real(calc_power(val, t_sample , 1/f_symbol)) for val in segment_output.values()]
    rest_xvals = [float(key)*rest_dz + int(full_distance / z_length) * z_length for key in segment_output.keys()]
    
    if DEBUG:
        print(f"last Segment ({rest_z_length} km) power values: {rest_power}")
    if DEBUG_PLOT:
        plt.plot(np.real(rest_output[f"{str(nz)}"]), label="Real")
        plt.plot(np.imag(rest_output[f"{str(nz)}"]), label="Imag")
        plt.legend()
        plt.show()
    
    power_full_distance += rest_power
    xvals_full_distance += rest_xvals


fig1, ax1 = plt.subplots(1, figsize=figure_size)

ax1.plot(xvals_full_distance, power_full_distance)
ax1.set_xlim(np.amin(xvals_full_distance), np.amax(xvals_full_distance))
ax1.set_ylim(bottom=0)


tikzplotlib.save('../../../bachelorarbeit-ausarbeitung/figures/plots/long_distance_power_level.tex', figure=fig1, figureheight="\\figheight", figurewidth="\\figwidth")

