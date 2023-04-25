#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Tue April 25 2023
#@author: Xian Wang

from __future__ import division
import numpy as np
import scipy.io as sio
import QOC_utils as utils
import os

cwd = os.getcwd()
os.chdir(cwd)

t_min = 0.0
t_max = 1000.0
tau = 0.05                #time step
t_state = np.arange(t_min, t_max + tau/2, tau, dtype=np.float64)            # time interval for states, [t_min, t_max]
t_field = np.arange(t_min + tau/2, t_max, tau, dtype=np.float64)            # time interval for pulses, [t_min + tau/2, t-max - tau/2]

Fs = 1 / tau              # resolution of FFT
N = len(t_state)
f = np.arange(0.0, N-1, 1.0, dtype=np.float64) * Fs / (N-1)    # frequency domain

alpha = 0.0               # penalty factor

#define all parameters above

n_qb = 6                  # number of qubits
init_state_idx = 0        # index of initial state in Q_dagger
target_state_idx = 3      # index of target state in Q_dagger

omega = 2.0               # amplitude of the static field along z-axis
coupling = 0.8            # coupling coefficient of qubits being positioned next to each other

fidelity = .999           # try to reach this fidelity
maxIter = 100             # upper limit of number of iterations
ifContinuedSolving = False

# load QOC optimization module:
qcm = utils.NIC_CAGE_qubit(t_state, t_field, tau, f, N, alpha, n_qb, init_state_idx, target_state_idx, omega, coupling, ifContinuedSolving)

random_seed_1 = 42          # comment the random seed when not testing for timing
Bx = qcm.init_Bf(random_seed_1)           # define initial Bx with white noise

random_seed_2 = 5           # comment the random seed when not testing for timing
By = qcm.init_Bf(random_seed_2)           # define initial By with white noise

init_state = qcm.init_state             # generating initial state
target_state = qcm.target_state         # generating target state

# optimization process:
Bx, By, final_state, J, P, Bx_amp_list, By_amp_list, grad_Bx_amp_list, grad_By_amp_list = qcm.optimize(Bx, By, fidelity, maxIter)     #psi_N, J, P omitted when not testing

# plot the output, comment when running on supercomputer:                          
qcm.plot(Bx, By, J, P, Bx_amp_list, By_amp_list, grad_Bx_amp_list, grad_By_amp_list, f_max = 1.0)

fft_Bx = qcm.solve_fft(Bx, N-1, N-1)    # FFT of B-fields, power spectra
fft_By = qcm.solve_fft(By, N-1, N-1)   
#prob = P[-1]                           # probability at last iteration

'''
curr_name = 'QC180_' + str(n_qb) + 'qb_cpl_decomposed.mat'        # save to this .mat file

sio.savemat(curr_name, {'init_state':init_state,
                         'target_state':target_state,
                         'final_state':final_state,
                         'init_state_idx':init_state_idx,
                         'target_state_idx':target_state_idx,
                         'n_qb':n_qb,
                         'J':J,
                         'P':P,
                         'omega':omega,
                         'coupling':coupling,
                         't_max':t_max,
                         'tau':tau,
                         'Fs':Fs,
                         'random_seed_1':random_seed_1,
                         'random_seed_2':random_seed_2,
                         't_state':t_state,
                         't_field':t_field,
                         'f':f,
                         'Bx':Bx,
                         'By':By,
                         'Bx_amp_list':Bx_amp_list,
                         'By_amp_list':By_amp_list,
                         'grad_Bx_amp_list':grad_Bx_amp_list,
                         'grad_By_amp_list':grad_By_amp_list,
                         'fft_Bx':fft_Bx,
                         'fft_By':fft_By,
                         'prob':prob})
'''
