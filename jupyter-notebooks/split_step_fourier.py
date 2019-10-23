#!/usr/bin/env python3

# Imports

import numpy as np


# Filter Definitions

# taken from https://github.com/kit-cel/lecture-examples/blob/master/nt1/vorlesung/3_mod_demod/pulse_shaping.ipynb
def get_rc_ir(syms, r, f_symbol, n_up):
    """Determines normed coefficients of an RC filter

    Formula out of: K.-D. Kammeyer, Nachrichtenübertragung
    At poles, l'Hospital was used

    :param syms: "normed" length of ir. ir-length will be 2*syms+1
    :param r: roll-off factor [Float]
    :param f_symbol: symbol rate [Baud]
    :param n_up: upsampling factor [Int]
    
    :returns: tuple containing time-index-array and impulse response in an array

    """
    
    # initialize output length and sample time
    T_symbol = 1.0 / f_symbol  # Duration of one Symbol
    t_sample = T_symbol / n_up  # length of one sample is the symbol-duration divided by the oversampling factor (=1/sampling rate)
    T_ir = 2 * syms * T_symbol  # Duration of the impulse response is positive and negative normed symbols added multplied by Symbol Duration
    ir = np.zeros(int(T_ir / t_sample) + 1)  # samples of impulse response is definied by duration of the ir divided by the sample time plus one for the 0th sample

    # time indices and sampled time
    k_steps = np.arange(- T_ir / t_sample / 2, T_ir / t_sample / 2 + 1, dtype=int)
    t_steps = k_steps * t_sample
    
    for k in k_steps:
        
        if t_steps[k] == 0:
            ir[ k ] = 1. / T_symbol
            
        elif r != 0 and np.abs( t_steps[k] ) == T_symbol / ( 2.0 * r ):
            ir[ k ] = r / ( 2.0 * T_symbol ) * np.sin( np.pi / ( 2.0 * r ) )
            
        else:
            ir[ k ] = np.sin(np.pi * t_steps[k] / T_symbol) / np.pi / t_steps[k] * np.cos(r * np.pi * t_steps[k] / T_symbol)                    / (1.0 - (2.0 * r * t_steps[k] / T_symbol)**2)
    
    ir /= np.linalg.norm(ir)
    
    return t_sample, ir


def get_rrc_ir(syms, r, f_symbol, n_up):
    """Determines normed coefficients of an RRC filter

    Formula out of: K.-D. Kammeyer, Nachrichtenübertragung
    At poles, l'Hospital was used

    :param syms: "normed" length of ir. ir-length will be 2*syms+1
    :param r: roll-off factor [Float]
    :param f_symbol: symbol rate [Baud]
    :param n_up: upsampling factor [Int]
    
    :returns: tuple containing time-index-array and impulse response in an array

    """
    
    # initialize output length and sample time
    T_symbol = 1.0 / f_symbol  # Duration of one Symbol
    t_sample = T_symbol / n_up  # length of one sample is the symbol-duration divided by the oversampling factor (=1/sampling rate)
    T_ir = 2 * syms * T_symbol  # Duration of the impulse response is positive and negative normed symbols added multplied by Symbol Duration
    ir = np.zeros(int(T_ir / t_sample) + 1)  # samples of impulse response is definied by duration of the ir divided by the sample time plus one for the 0th sample

    # time indices and sampled time
    k_steps = np.arange(- T_ir / t_sample / 2, T_ir / t_sample / 2 + 1, dtype=int)
    t_steps = k_steps * t_sample
    
    for k in k_steps.astype(int):

        if t_steps[k] == 0:
            ir[ k ] = (np.pi + 4.0 * r - np.pi * r) / (np.pi * T_symbol)

        elif r != 0 and np.abs(t_steps[k] ) == T_symbol / ( 4.0 * r ):
            ir[ k ] = r * (-2.0 * np.cos(np.pi * (1.0 + r) / (4.0 * r)) + np.pi * np.sin(np.pi * (1.0 + r) / (4.0 * r))) / (np.pi * T_symbol)

        else:
            ir[ k ] = ( 4.0 * r * t_steps[k] / T_symbol * np.cos(np.pi * (1.0 + r) * t_steps[k] / T_symbol) + np.sin(np.pi * (1.0 - r) * t_steps[k] / T_symbol))                    / (( 1.0 - (4.0 * r * t_steps[k] / T_symbol)**2) * np.pi * t_steps[k])

    ir /= np.linalg.norm(ir)

    return t_sample, ir


def get_gaussian_ir(syms, f_symbol, n_up):
    """Determines normed coefficients of an Gaussian filter

    :param syms: "normed" length of ir. ir-length will be 2*syms+1
    :param f_symbol: symbol rate [Baud]
    :param n_up: upsampling factor [Int]
    
    :returns: tuple containing time-index-array and impulse response in an array

    """
    
    # integral -t/2 bis t/2 e^-x^2/kappa dz = 0,95*gesamtleistung
    # initialize sample time
    T_symbol = 1.0 / f_symbol  # Symbol time; in this case = pulse length
    t_sample = T_symbol / n_up  # length of one sample is the symbol-duration divided by the oversampling factor (=1/sampling rate)
    T_ir = 2 * syms * T_symbol  # Duration of the impulse response is positive and negative normed symbols added multplied by Symbol Duration
    r = 2.57583 / T_symbol
    
    # time indices and sampled time
    k_steps = np.arange(- T_ir / t_sample / 2, T_ir / t_sample / 2 + 1, dtype=int)
    t_steps = k_steps * t_sample

    ir = (r / np.sqrt(np.pi)) * np.exp(-np.square(r * t_steps))    
    ir /= np.linalg.norm(ir)

    return t_sample, ir


def get_gaussian_ir2(syms, r, f_symbol, n_up):
    """Determines normed coefficients of an Gaussian filter

    :param syms: "normed" length of ir. ir-length will be 2*syms+1
    :param r: roll-off factor [Float]
    :param f_symbol: symbol rate [Baud]
    :param n_up: upsampling factor [Int]
    
    :returns: tuple containing time-index-array and impulse response in an array

    """
    
    # integral -t/2 bis t/2 e^-x^2/kappa dz = 0,95*gesamtleistung
    # initialize sample time
    T_symbol = 1.0 / f_symbol  # Symbol time; in this case = pulse length
    t_sample = T_symbol / n_up  # length of one sample is the symbol-duration divided by the oversampling factor (=1/sampling rate)
    T_ir = 2 * syms * T_symbol  # Duration of the impulse response is positive and negative normed symbols added multplied by Symbol Duration
    print(r)
    
    # time indices and sampled time
    k_steps = np.arange(- T_ir / t_sample / 2, T_ir / t_sample / 2 + 1, dtype=int)
    t_steps = k_steps * t_sample

    ir = (r / np.sqrt(np.pi)) * np.exp(-np.square(r * t_steps))    
    ir /= np.linalg.norm(ir)

    return t_sample, ir


# Pulse forming

def generate_signal(modulation, data, pulse, syms):
    """Generates send Signal
    
    This function calculates send signal with variable 
    
    NOTE: If pulse doesn't meet the nyquist isi criterion set syms = 0.

    :param modulation: Dict mapping symbols to send-values. Ie {00: 1+1j, 01: 1-1j, 11: -1-1j, 10: -1+1j} 
    :param data: Data to send. Should be an array containing the symbols.
    :param pulse: Impulse response of pulse filter
    :param syms: "Normed" symbol length of pulse
    
    :returns: array conatining send signal

    """

    assert isinstance(pulse, np.ndarray), "Pulse should be an numpy array"
    assert isinstance(data, (list, np.ndarray)), "Send data should be a list or numpy array"
    assert syms >= 0 and isinstance(syms, int), "syms should be positive int or zero"

    send_symbols = [modulation[str(symbol)] for symbol in data]

    if syms == 0:
        send_symbols_up = np.zeros(len(data) * pulse.size)
        send_symbols_up[ : : pulse.size] = send_symbols
    else:
        n_up = int((pulse.size - 1) / (2 * syms))
        send_symbols_up = np.zeros(len(data) * n_up)
        send_symbols_up[ : : n_up] = send_symbols
    
    send_signal = np.convolve(pulse, send_symbols_up)

    return send_signal


# SplitStep Fourier Function

def splitstepfourier(u0, dt, dz, nz, alpha, beta2, gamma):
    """Split-step fourier method
    
    This function solves the nonlinear Schrodinger equation for pulse propagation in an optical fiber using the split-step Fourier method.
    Python-Implementation of the Matlab-Code from "SSPROP" found here: https://www.photonics.umd.edu/software/ssprop/
    
    NOTE: Dimensions / Units of the params can be anything. They just have to be consistent.

    :param u0: input signal (array)
    :param dt: time step between samples (sample time)
    :param dz: propagation stepsize (delta z)
    :param nz: number of steps. ie totalsteps = nz * dz
    :param alpha: power loss coeficient
    :param beta2: dispersion polynomial coefficient
    :param gamma: nonlinearity coefficient
    
    :returns: signal at the output

    """
    assert isinstance(u0, np.ndarray), "Input signal should be a numpy array."

    nt = len(u0)
    dw = 2 * np.pi * np.fft.fftfreq(nt,dt)

    # Linear operator (frequency domain)
    linear_operator = np.exp((-alpha/2 - 1j * beta2 / 2 * np.square(dw)) * dz)
    linear_operator_halfstep = np.exp((-alpha/2 - 1j * beta2 / 2 * np.square(dw)) * dz / 2)

    # Nonlinear operator (time domain)
    nonlinear_operator = lambda u : np.exp(-1j * gamma * np.square(np.absolute(u)) * dz)

    start = u0
    # Start (half linear step)
    f_temp = np.fft.fft(start)
    f_temp = f_temp * linear_operator_halfstep
    # First Nonlinear step
    temp = np.fft.ifft(f_temp)
    temp = temp * nonlinear_operator(temp)
    
    # Main Loop (nz-1 * (full linear steps + nonlinear steps))
    for step in range(1,nz):
        # Full linear step
        f_temp = np.fft.fft(temp)
        f_temp = f_temp * linear_operator
        # Nonlinear Step
        temp = np.fft.ifft(f_temp)
        temp = temp * nonlinear_operator(temp)
    
    # End (half linear step)
    f_temp = np.fft.fft(temp)
    f_end = f_temp * linear_operator_halfstep
    
    output = np.fft.ifft(f_end)
    
    return output,linear_operator,nonlinear_operator(start)

