#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/split_step_fourier.py")

import split_step_fourier

# showing figures inline
get_ipython().run_line_magic('matplotlib', 'inline')
# plotting options 
font = {'size'   : 20}
plt.rc('font', **font)
plt.rc('text', usetex=True)


# parameters of the filters
f_symbol = 2.0  # sample rate (Baud)
n_up = 100 # samples per symbol (>1 => oversampling)

r_rc = .33
r_rrc = .33
r_gaussian = 0.8

syms_per_filt = 10  # symbols per filter (plus minus in both directions)

t_rc, rc = split_step_fourier.get_rc_ir( syms_per_filt, r_rc, f_symbol, n_up )
t_rrc, rrc = split_step_fourier.get_rrc_ir( syms_per_filt, r_rrc, f_symbol, n_up )
t_gaussian, gaussian = split_step_fourier.get_gaussian_ir( syms_per_filt, r_gaussian, f_symbol, n_up )




matplotlib.rc('figure', figsize=(24, 12) )

plt.plot( t_rc, rc, linewidth=2.0, label='RC' )
plt.plot( t_rrc, rrc, linewidth=2.0, label='RRC' )
plt.plot( t_gaussian, gaussian, linewidth=2.0, label='Gaussian' )

plt.grid( True )
plt.legend( loc='upper right' )
plt.title( 'Impulse Responses' )


# Comparison of convolved rrc with rc

t_rc, rc = split_step_fourier.get_rc_ir( syms_per_filt, r_rc, f_symbol, n_up )
t_rrc, rrc = split_step_fourier.get_rrc_ir( syms_per_filt, r_rrc, f_symbol, n_up )

rrc_convolved = np.convolve(rrc, rrc, mode='same')
rrc_convolved /= np.linalg.norm(rrc_convolved)

matplotlib.rc('figure', figsize=(24, 12) )

plt.plot( rc, linewidth=2.0, label='RC' )
plt.plot( rrc_convolved, linewidth=2.0, label='Convolved RRC')

plt.grid( True )
plt.legend( loc='upper right' )
plt.title( 'Impulse Responses' )




