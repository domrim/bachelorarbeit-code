#!/usr/bin/env python3

# Imports
import numpy as np


# System parameters
D = 17  # ps/nm/km
gamma = 1.3e-3  # 1/W/m
B_signal = 30e9  # signal bandwidth = channel spacing for Nyquist WDM [Hz]
N_channel = 17  # numbers of channels
NFdB = 4  # amplifier noise figure [dB]
Length=100e3  # span length (m)
alpha=0.2e-3  # fibre attenuation (dB/m)
N_spans=10  # number of spans
N_pol=2  # number of polarizations


# Constants
h = 6.6256e-34  # Planck's constant [J/s]
nu = 299792458 / 1550e-9  # light frequency [Hz]
SNR2OSNR = 10 * np.log10( B_signal / 12.5e9 * N_pol / 2 )  # Eq. (34) of Essiambre et al., "Capacity Limits of Optical Fiber Networks," J. Light. Technol., vol. 28, no. 4, pp. 662-701, Feb. 2010.
dB2Neper = 10 / np.log(10)


# Set launch power per channel
vec_start = -10
vec_stop = 10
vec_step = .25
powerVec = np.linspace(-10, 10, num=int(( vec_stop-vec_start) / vec_step + 1 ))  # launch power [dBm]


# Some more quantities
B_total = N_channel * B_signal  # total system bandwidth
beta2 = -np.square(1550e-9) * ( D * 1e-6 ) / ( 2 * np.pi * 3e8)  # propagation constant
GaindB = Length * alpha  # amplifier gain (dB)
L_eff = (( 1 - np.exp ( -( alpha / dB2Neper ) * Length )) / (alpha / dB2Neper))  # effective length [m]
L_effa = 1 / (alpha/dB2Neper)  # asymptotic effective length [m]
G_tx_ch = np.divide( np.power( 10, ( powerVec - 30 ) / 10 ) , B_signal) # [W/Hz]
ASE = N_pol * N_spans * np.power(10, NFdB / 10 ) / 2 * ( np.power(10, GaindB / 10 ) - 1 ) * h * nu  # Eq. (50)


# GN model
epsilon = 3 / 10 * np.log( 1 + 6 / Length * L_effa / ( np.arcsinh( 0.5 * np.square(np.pi) * np.abs( beta2 ) * L_effa * np.square( B_total ))))  # Eq. (40)
G_NLI = np.multiply( np.square(gamma) , np.power( G_tx_ch, 3 ) * np.square(L_eff) * np.power( 2/3, 3 ) * np.arcsinh( 0.5 * np.square( np.pi ) * np.abs( beta2 ) * L_effa * np.square(B_total))/( np.pi * np.abs( beta2 ) * L_effa))  # Eq. (36)


# SNR calculation
GN = {}

GN['SNR_NLI'] = 10 * np.log10( np.divide( G_tx_ch , ( ASE + np.power( N_spans, ( 1 + epsilon )) * G_NLI) ))  # Eq. (16), (22)
GN['SNR_ASE'] = 10 * np.log10( np.divide( G_tx_ch, ASE ))


# OSNR calculation
GN['OSNR_NLI'] = np.add(GN['SNR_NLI'], SNR2OSNR)
GN['OSNR_EDFA'] = np.add(GN['SNR_ASE'], SNR2OSNR)

GN['power'] = powerVec


for key, item in GN.items():
    print("{}:\n{}".format(key, item))




