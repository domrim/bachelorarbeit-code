#!/usr/bin/env python3

# Imports

import numpy as np


# Parameters

# modulation scheme and constellation points
M = 2
constellation_points = [ -1, 1 ]

t_symbol = 1.0  # symbol time
n_symbol = 100  # number of symbols

n_up = 8 # samples per symbol (>1 => oversampling)


# Filter Definitions

# Raised-Cosine
# find impulse response of an RC filter
# taken from https://github.com/kit-cel/lecture-examples/blob/master/nt1/vorlesung/3_mod_demod/pulse_shaping.ipynb
def get_rc_ir(K, n_up, t_symbol, r):
    
    ''' 
    Determines coefficients of an RC filter 
    
    Formula out of: K.-D. Kammeyer, Nachrichtenübertragung
    At poles, l'Hospital was used 
    
    NOTE: Length of the IR has to be an odd number
    
    IN: length of IR (K), upsampling factor (n_up), symbol time (t_symbol), roll-off factor (r)
    OUT: filter coefficients (rc)
    '''

    # check that IR length is odd
    assert K % 2 == 1, 'Length of the impulse response should be an odd number'
    
    # map zero r to close-to-zero
    #if r == 0:
    #    r = 1e-32
    
    # initialize output length and sample time
    rc = np.zeros( K )
    t_sample = t_symbol / n_up
    
    # time indices and sampled time
    k_steps = np.arange( -(K-1) / 2.0, (K-1) / 2.0 + 1 )   
    t_steps = k_steps * t_sample
    
    for k in k_steps.astype(int):
        
        if t_steps[k] == 0:
            rc[ k ] = 1. / t_symbol
            
        elif r != 0 and np.abs( t_steps[k] ) == t_symbol / ( 2.0 * r ):
            rc[ k ] = r / ( 2.0 * t_symbol ) * np.sin( np.pi / ( 2.0 * r ) )
            
        else:
            rc[ k ] = np.sin( np.pi * t_steps[k] / t_symbol ) / np.pi / t_steps[k]                 * np.cos( r * np.pi * t_steps[k] / t_symbol )                 / ( 1.0 - ( 2.0 * r * t_steps[k] / t_symbol )**2 )
 
    return rc


def get_rrc_ir(K, n_up, t_symbol, r):
    
    # check that IR length is odd
    assert K % 2 == 1, 'Length of the impulse response should be an odd number'
    
    # initialize output length and sample time
    rrc = np.zeros( K )
    t_sample = t_symbol / n_up
    
    # time indices and sampled time
    k_steps = np.arange( -(K-1) / 2.0, (K-1) / 2.0 + 1 )   
    t_steps = k_steps * t_sample
    
    for k in k_steps.astype(int):

        if t_steps[ k ] == 0:
            rrc[ k ] = 1.0 - r + ( 4.0 * r / np.pi )
            
        elif r != 0 and np.abs( t_steps[k] ) == t_symbol / ( 4.0 * r ):
            rrc[ k ] = 0 #TODO
        
        else:
            rrc[ k ] = 0 #TODO
    
    return rrc


def get_gaussian_ir(K, n_up, t_symbol, r):
    
    # check that IR length is odd
    assert K % 2 == 1, 'Length of the impulse response should be an odd number'
    
    # initialize sample time
    t_sample = t_symbol / n_up
    
    # time indices and sampled time
    k_steps = np.arange( -(K-1) / 2.0, (K-1) / 2.0 + 1 )   
    t_steps = k_steps * t_sample
    
    gaussian = ( r / np.sqrt( np.pi )) * np.exp( -np.square( r * t_steps ))
    
    return gaussian


# Tests

# parameters of the filters
r_rc = .33
r_rrc = .33
r_gaussian = 0.8

syms_per_filt = 4  # symbols per filter (plus minus in both directions)
K_filt = 2 * syms_per_filt * n_up + 1         # length of the fir filter

rc = get_rc_ir( K_filt, n_up, t_symbol, r_rc )
#rc /= np.linalg.norm( rc ) 

gaussian = get_gaussian_ir( K_filt, n_up, t_symbol, r_gaussian )
#gaussian /= np.linalg.norm( gaussian )


import matplotlib.pyplot as plt
import matplotlib

# showing figures inline
get_ipython().run_line_magic('matplotlib', 'inline')
# plotting options 
font = {'size'   : 20}
plt.rc('font', **font)
plt.rc('text', usetex=True)

matplotlib.rc('figure', figsize=(18, 10) )

plt.plot( np.arange( np.size( rc ) ) * t_symbol / n_up, rc, linewidth=2.0, label='RC' )
plt.plot( np.arange( np.size( gaussian ) ) * t_symbol / n_up, gaussian, linewidth=2.0, label='Gaussian' )

plt.ylim( (-.3, 1.1 ) ) 
plt.grid( True )
plt.legend( loc='upper right' )
plt.title( 'Impulse Responses' )




