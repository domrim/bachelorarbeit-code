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
    :param r: roll-off factor
    :param f_symbol: symbol rate
    :param n_up: upsampling factor
    
    :returns: tuple containing time-index-array and impulse response in an array

    """
    
    # initialize output length and sample time
    ir = np.zeros( 2 * syms * n_up + 1 )
    T_symbol = 1.0 / f_symbol
    t_sample = T_symbol / n_up
    
    # time indices and sampled time
    k_steps = np.arange( - syms * n_up, syms * n_up + 1 )   
    t_steps = k_steps * t_sample
    
    for k in k_steps.astype(int):
        
        if t_steps[k] == 0:
            ir[ k ] = 1. / T_symbol
            
        elif r != 0 and np.abs( t_steps[k] ) == T_symbol / ( 2.0 * r ):
            ir[ k ] = r / ( 2.0 * T_symbol ) * np.sin( np.pi / ( 2.0 * r ) )
            
        else:
            ir[ k ] = np.sin( np.pi * t_steps[k] / T_symbol ) / np.pi / t_steps[k]                 * np.cos( r * np.pi * t_steps[k] / T_symbol )                 / ( 1.0 - ( 2.0 * r * t_steps[k] / T_symbol )**2 )
    
    ir /= np.linalg.norm(ir)
    
    return t_sample, ir


def get_rrc_ir(syms, r, f_symbol, n_up):
    """Determines normed coefficients of an RRC filter

    Formula out of: K.-D. Kammeyer, Nachrichtenübertragung
    At poles, l'Hospital was used

    :param syms: "normed" length of ir. ir-length will be 2*syms+1
    :param r: roll-off factor
    :param f_symbol: symbol rate
    :param n_up: upsampling factor
    
    :returns: tuple containing time-index-array and impulse response in an array

    """
    
    # initialize output length and sample time
    ir = np.zeros( 2 * syms * n_up + 1 )
    T_symbol = 1.0 / f_symbol
    t_sample = T_symbol / n_up
    
    # time indices and sampled time
    k_steps = np.arange( - syms * n_up, syms * n_up + 1 )   
    t_steps = k_steps * t_sample
    
    for k in k_steps.astype(int):

        if t_steps[k] == 0:
            ir[ k ] = ( np.pi + 4.0 * r - np.pi * r ) / ( np.pi * T_symbol)

        elif r != 0 and np.abs(t_steps[k] ) == T_symbol / ( 4.0 * r ):
            ir[ k ] = r * ( -2.0 * np.cos( np.pi * ( 1.0 + r ) / ( 4.0 * r ) )                             + np.pi * np.sin( np.pi * ( 1.0 + r ) / ( 4.0 * r ) ) )             / ( np.pi * T_symbol )

        else:
            ir[ k ] = ( 4.0 * r * t_steps[k] / T_symbol                         * np.cos( np.pi * ( 1.0 + r ) * t_steps[k] / T_symbol)                         + np.sin( np.pi * (1.0 - r ) * t_steps[k] / T_symbol ))             / (( 1.0 - ( 4.0 * r * t_steps[k] / T_symbol)**2 ) * np.pi * t_steps[k])

    ir /= np.linalg.norm(ir)

    return t_sample, ir


def get_gaussian_ir(syms, r, f_symbol, n_up):
    """Determines normed coefficients of an Gaussian filter

    :param syms: "normed" length of ir. ir-length will be 2*syms+1
    :param r: roll-off factor
    :param f_symbol: symbol rate
    :param n_up: upsampling factor
    
    :returns: tuple containing time-index-array and impulse response in an array

    """
    
    # initialize sample time
    T_symbol = 1.0 / f_symbol
    t_sample = T_symbol / n_up
    
    # time indices and sampled time
    k_steps = np.arange( - syms * n_up, syms * n_up + 1 )   
    t_steps = k_steps * t_sample
    
    ir = ( r / np.sqrt( np.pi )) * np.exp( -np.square( r * t_steps ))    
    ir /= np.linalg.norm(ir)

    return t_sample, ir


# Pulse forming

def generate_signal(modulation, data, pulse, syms):
    """Generates send Signal
    
    This function calculates send signal with variable 
    
    NOTE: If pulse doesn't meet the nyquist isi criterion hand syms = 0 to the function.

    :param modulation: Dict mapping symbols to send-values. Ie {00: 1+1j, 01: 1-1j, 11: -1-1j, 10: -1+1j} 
    :param data: Data to send. Should be an array containing the symbols.
    :param pulse: Impulse response of pulse filter
    :param syms: "Normed" symbol length of pulse
    
    :returns: array conatining send signal

    """

    assert isinstance(pulse, np.ndarray), "Pulse should be an numpy array"
    assert isinstance(data, (list, np.ndarray)), "Send data should be a list or numpy array"
    assert syms >= 0 and isinstance(syms, int), "syms should be positive int or zero"

    send_symbols = [ modulation[str(symbol)] for symbol in data ]

    if syms == 0:
        send_symbols_up = np.zeros( len(data) * pulse.size )
        send_symbols_up[ : : pulse.size ] = send_symbols
    else:
        n_up = int((pulse.size - 1) / (2 * syms))
        send_symbols_up = np.zeros( len(data) * n_up )
        send_symbols_up[ : : n_up ] = send_symbols
    
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
    dw = 2 * np.pi * np.append(np.arange(nt/2), np.arange(-nt/2, 0)) / (dt * nt)

    # Linear operator (frequency domain)
    linear_operator = np.exp((-alpha/2 - 1j * beta2 / 2 * np.square(dw)) * dz)
    linear_operator_halfstep = np.exp((-alpha/2 - 1j * beta2 / 2 * np.square(dw)) * dz / 2)

    # Nonlinear operator (time domain)
    nonlinear_operator = lambda u : np.exp(-1j * gamma * np.square(np.absolute(u)) * dz)

    # Start (half linear step)
    start = u0
    f_temp = np.fft.fft(start)
    f_temp = f_temp * linear_operator_halfstep
    
    # Main Loop (nz-1 nonlinear + full linear steps)
    for step in range(1,nz-1):
        temp = np.fft.ifft(f_temp)
        temp = temp * nonlinear_operator(start)
        f_temp = np.fft.fft(temp)
        f_temp = f_temp * linear_operator
    
    # End (nonlinear + half linear step)
    temp = np.fft.ifft(f_temp)
    temp = temp * nonlinear_operator(temp)
    f_temp = np.fft.fft(temp)
    f_end = f_temp * linear_operator_halfstep
    
    output = np.fft.ifft(f_end)
    
    return output

