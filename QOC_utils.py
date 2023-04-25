#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Tue April 25 2023
#@author: Xian Wang 


from __future__ import division
import numpy as np
import scipy.io as sio
import copy
from scipy.special import comb
from scipy.linalg import solve_banded, expm    #, solve
import time
from scipy.fftpack import fft, ifft
from math import pi, factorial, floor   #, ceil, log
import warnings
import matplotlib.pyplot as plt
import cProfile, pstats, io

def profile(fnc):
    #A decorator that uses cProfile to profile a function

    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    
    return inner

class QOC_qubit:
    def __init__(self, t_state, t_field, tau, f, N, alpha, n_qb, init_state_idx, target_state_idx, omega, coupling, ifContinuedSolving):
        self.t_state = t_state         # time interval for states
        self.t_field = t_field         # time interval for pulses
        self.tau = tau                 # time step
        self.f = f                     # frequency domain
        self.N = N                     # number of time steps
        self.alpha = alpha             # penalty factor
        self.ifContinuedSolving = ifContinuedSolving         # if loading output of previous optimization
        self.n_qb = n_qb               # number of qubits
        self.omega = omega             # amplitude of the static field along z-axis
        self.coupling = coupling       # coupling coefficient of qubits being positioned next to each other
        
        # define spin-up, spin down states, Pauli matrices, Identity-2 and other gate operations:
        self.spinup = np.array([1.0, 0.0], dtype=np.float64)
        self.spindown = np.array([0.0, 1.0], dtype=np.float64)
        self.sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        self.sigmax = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        self.sigmay = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        self.idn = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        self.sqrtX = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=np.complex128)
        self.hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)
        self.phaseS = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
        self.phaseT = np.array([[1.0, 0.0], [0.0, (1.0 + 1.0j) / np.sqrt(2.0)]], dtype=np.complex128)
        
        # define the drift Hamiltonian Hz and H_coupling, and control Hamiltonian (dipole moment) Hx and Hy:
        self.Hz = self.solve_for_H_Sn(self.n_qb, self.sigmaz, eval_complex = False)
        self.Hz_coupling = self.solve_for_H_coupling_ring(self.n_qb, self.sigmaz, eval_complex = False)
        #self.Hz_coupling = self.solve_for_H_coupling_lattice(self.n_qb, self.dim_lattice, self.sigmaz, eval_complex = False)
        self.H0 = - self.omega * 0.5 * self.Hz + self.coupling * 0.25 * self.Hz_coupling
        self.H0_diag = np.diag(self.H0)
        self.Hx = 0.5 * self.solve_for_H_Sn(self.n_qb, self.sigmax, eval_complex = False)
        self.Hy = 0.5 * self.solve_for_H_Sn(self.n_qb, self.sigmay, eval_complex = True)
        #self.Hx = self.solve_for_H_coupling_ring(self.n_qb, self.sigmax, eval_complex = False)
        #self.Hy = self.solve_for_H_coupling_ring(self.n_qb, self.sigmay, eval_complex = True)
        #self.idn_nqb_full = np.identity(int(2**self.n_qb), dtype=np.float64)
        #self.idn_nqb_diag_full = np.diag(self.idn_nqb)
        
        # Need to choose Sn or Dn method here:
        # define the adjoint matrix Q for transformation with |J, M> basis, record J, M for each subspace and eigenstate:
        # for Sn symmetry only:
        #self.Q_full, self.Q_dagger_full, self.J_subspace, self.J_eigenstate, self.M_eigenstate = self.solve_for_Adjoint_Q_CG_method(self.n_qb)
        #self.Q_full, self.Q_dagger_full, self.J_subspace, self.J_eigenstate, self.M_eigenstate = self.Young_method_Sn(self.n_qb)
        
        # for Dn symmetry only:
        self.Q_full, self.Q_dagger_full, self.dimension_subspace, self.M_eigenstate, self.basis_split_number = self.diag_operator_method_Dn(self.n_qb)
        #print(self.dimension_subspace)
        #print(self.basis_split_number)
        
        # check if there is coupling among subspaces and |J, M> eigenstates
        # for Sn symmetry only:
        #self.sel_mat_subspace, self.sel_mat_eigenstate, self.M_notConserved_subspace = self.solve_for_selection_mat(self.H_coupling, self.Q_full, self.Q_dagger_full, self.J_subspace, self.n_qb)
        #print(self.sel_mat_subspace)
        #print(self.sel_mat_eigenstate)
        #print(self.M_notReserved_subspace)
        
        # define initial and target states with indices
        #self.H_hadamard = self.solve_for_quantum_gate(self.n_qb, self.hadamard)
        #self.init_state = np.matmul(self.H_hadamard, self.Q_dagger_full[0])
        self.init_state = self.Q_dagger_full[0]
        #self.target_state = self.Q_dagger_full[self.n_qb]
        self.target_state = self.Q_dagger_full[ int(self.dimension_subspace[0] - 1) ]
        #self.H_phaseT = self.solve_for_quantum_gate(self.n_qb, self.phaseT)
        #self.target_state = np.matmul(self.H_phaseT, self.init_state)
        
        # Need to define Q here, whether to use original H, original Q or reduced Q:
        # keep original H, make Q identity:
        #self.Q, self.Q_dagger, self.dim_H, self.n_subdiag = self.keep_original_H(self.Q_dagger_full, self.n_qb)
        
        # keep original Q:
        #self.Q, self.Q_dagger, self.dim_H, self.n_subdiag = self.keep_original_Q(self.Q_dagger_full, self.n_qb)
        
        # reduce Q with essential component analysis, based on |J, M> basis, only applies when there is no coupling term or there is all-to-all coupling term:
        # for Sn symmetry only:
        #self.Q, self.Q_dagger, self.dim_H, self.n_subdiag = self.solve_for_ECA_Q_Sn(self.Q_dagger_full, self.J_subspace, self.init_state, self.target_state, self.n_qb)
        
        # reduce Q with essential component analysis:
        # for Dn symmetry only:
        self.Q, self.Q_dagger, self.dim_H = self.solve_for_ECA_Q_Dn(self.Q_dagger_full, self.dimension_subspace, self.init_state, self.target_state, self.n_qb)
        
        # solve for n_subdiag for Q_dagger_Hx_Q:
        self.n_subdiag = self.solve_for_n_subdiag(self.dim_H, np.matmul(np.matmul(self.Q_dagger, self.Hx), self.Q) )
        #print(self.n_subdiag)
        
        # reduce Q with user-defined |J, M> basis, output dimension of H and number of sub-diagonals:
        #self.Q, self.Q_dagger, self.dim_H, self.n_subdiag = self.customize_Q(self.Q_dagger_full, self.n_qb)
        
        # rearrange Q by M-clustering:
        #self.Q, self.Q_dagger, self.dim_H, self.n_subdiag = self.rearrange_by_M_Q(self.Q_dagger_full, self.n_qb, self.M_eigenstate)
        
        self.idn_nqb = np.identity((self.dim_H), dtype=np.float64)
        self.idn_nqb_diag = np.diag(self.idn_nqb)
        
        # generate transformed H0, Hx, Hy and main diagonal of U with reduced Q
        self.mat_U_diag, self.Q_dagger_H0_Q, self.Q_dagger_Hx_Q, self.Q_dagger_Hy_Q = self.transform_Hamiltonian(self.Q, self.Q_dagger, self.dim_H)  #self.n_subdiag 
        
        # generate upper and lower bands of reduced H0, Hx, Hy:
        self.H0_ub, self.H0_lb = self.extract_band(self.Q_dagger_H0_Q, self.dim_H, self.n_subdiag, eval_Hy = False)
        self.Hx_ub, self.Hx_lb = self.extract_band(self.Q_dagger_Hx_Q, self.dim_H, self.n_subdiag, eval_Hy = False)
        self.Hy_ub, self.Hy_lb = self.extract_band(self.Q_dagger_Hy_Q, self.dim_H, self.n_subdiag, eval_Hy = True)
    
    def fwd_propagation_original(self, n_qb, omega, Bx, By, Hz, Hx, Hy, unitary):
    # Foward propagation of n-qubit system with original Hamiltonian
        tau = self.tau
        N = self.N
        
        H0 = 0.5 * omega * np.sum(Hz, axis=0)
        unitary_list = np.zeros(( int(((N-1)/100) + 1), 2**n_qb, 2**n_qb), dtype=np.complex128)
        unitary_list[0] = copy.deepcopy(unitary)
        
        # solve for unitary
        for j in range(0, N-1):
            Hx_sum = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
            Hy_sum = np.zeros((2**n_qb, 2**n_qb), dtype=np.complex128)  
            for k in range(0, n_qb):
                Hx_sum =  Hx_sum + Hx[k] * Bx[k, j]
                Hy_sum =  Hy_sum + Hy[k] * By[k, j]
            Hamiltonian = H0 + 0.5 * Hx_sum + 0.5 * Hy_sum
            propagator_U = expm(- 1j * tau * Hamiltonian )
            unitary = np.matmul(propagator_U, unitary)
            if ((j+1) % 100) == 0 :
                unitary_list[int((j+1) / 100)] = copy.deepcopy(unitary)
                
        return unitary, unitary_list
        
    def fwd_propagation_transformed(self, n_qb, omega, Bx, By, Hz, Hx, Hy, unitary):
    # Foward propagation of n-qubit system with transformed Hamiltonian
        tau = self.tau
        N = self.N
        
        #_, Q_dagger_nm1, J_subspace_nm1, _, _ = self.solve_for_Adjoint_Q_CG_method(n_qb-1)
        Q_dagger_Hz_Q = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        Q_dagger_Hx_Q = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        Q_dagger_Hy_Q = np.zeros((2**n_qb, 2**n_qb), dtype=np.complex128)
        Q_list = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
        Q_dagger_list = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
        for i in range(0, n_qb):
            Q_dagger_switch, Q_switch = self.index_switch(n_qb, i, int(n_qb-1) )
            #Q_dagger_dp, Q_dp = self.direct_product_single_qubit_S_n_minus_1(n_qb, Q_dagger_nm1, J_subspace_nm1)
            Q_dagger_list[i] = copy.deepcopy(Q_dagger_switch)
            Q_list[i] = copy.deepcopy(Q_switch)
        
        Q_dagger_Hz_Q = copy.deepcopy(Hz[n_qb-1])
        Q_dagger_Hx_Q = copy.deepcopy(Hx[n_qb-1])
        Q_dagger_Hy_Q = copy.deepcopy(Hy[n_qb-1])
                
        Q_dagger_H0_Q = 0.5 * omega * Q_dagger_Hz_Q
        
        #dim_subspace_list = np.int64(2 * (2 * J_subspace_nm1 + 1) )  # dimension of subspace in H(2^n, C)
        
        Q_dagger_H0_Q_block = np.zeros((2, 2), dtype=np.float64)
        Q_dagger_Hx_Q_block = np.zeros((2, 2), dtype=np.float64)
        Q_dagger_Hy_Q_block = np.zeros((2, 2), dtype=np.complex128)
        mesh_seq = np.arange(0, 2, 1)
        block_idx = np.ix_(mesh_seq, mesh_seq)
        Q_dagger_H0_Q_block = Q_dagger_H0_Q[block_idx]
        Q_dagger_Hx_Q_block = Q_dagger_Hx_Q[block_idx]
        Q_dagger_Hy_Q_block = Q_dagger_Hy_Q[block_idx]
        
        unitary_list = np.zeros(( int(((N-1)/100) + 1), 2**n_qb, 2**n_qb), dtype=np.complex128)
        unitary_list[0] = copy.deepcopy(unitary)
        
        # solve for unitary
        for j in range(0, N-1):
            propagator_U = np.identity((2**n_qb), dtype=np.complex128) 
            for i in range(0, n_qb):
                propagator_U_i = np.zeros((2**n_qb, 2**n_qb), dtype=np.complex128)
                idx = 0
                Hamiltonian_i_k = Q_dagger_H0_Q_block + 0.5 * Bx[i, j] * Q_dagger_Hx_Q_block + 0.5 * By[i, j] * Q_dagger_Hy_Q_block
                propagator_U_i_k = expm(- 1j * tau * Hamiltonian_i_k )   # to be block diagonalized
                for k in range(0, int(2**(n_qb - 1)) ):
                    mesh_seq = np.arange(idx, (idx + 2 ), 1)
                    block_idx = np.ix_(mesh_seq, mesh_seq)
                    propagator_U_i[block_idx] = copy.deepcopy(propagator_U_i_k )
                    idx = idx + 2
                propagator_U_i = np.matmul(np.matmul(Q_list[i], propagator_U_i), Q_dagger_list[i])
                propagator_U = np.matmul(propagator_U_i, propagator_U)
            unitary = np.matmul(propagator_U, unitary)
            if ((j+1) % 100) == 0 :
                unitary_list[int((j+1) / 100)] = copy.deepcopy(unitary)
                
        return unitary, unitary_list
    
    def fwd_propagation_cpl_original(self, n_qb, omega, coupling, Bx, By, Hz, Hx, Hy, H_cpl, unitary):
    # Foward propagation of n-qubit system with original Hamiltonian
        tau = self.tau
        N = self.N
        
        H0 = 0.5 * omega * np.sum(Hz, axis=0) + 0.25 * coupling * np.sum(H_cpl, axis=0)
        unitary_list = np.zeros(( int(((N-1)/100) + 1), 2**n_qb, 2**n_qb), dtype=np.complex128)
        unitary_list[0] = copy.deepcopy(unitary)
        
        # solve for unitary
        for j in range(0, N-1):
            Hx_sum = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
            Hy_sum = np.zeros((2**n_qb, 2**n_qb), dtype=np.complex128)  
            for k in range(0, n_qb):
                Hx_sum =  Hx_sum + Hx[k] * Bx[k, j]
                Hy_sum =  Hy_sum + Hy[k] * By[k, j]
            Hamiltonian = H0 + 0.5 * Hx_sum + 0.5 * Hy_sum
            propagator_U = expm(- 1j * tau * Hamiltonian )
            unitary = np.matmul(propagator_U, unitary)
            if ((j+1) % 100) == 0 :
                unitary_list[int((j+1) / 100)] = copy.deepcopy(unitary)
                
        return unitary, unitary_list
        
    def fwd_propagation_cpl_transformed(self, n_qb, omega, coupling, Bx, By, Hz, Hx, Hy, H_cpl, unitary):
    # Foward propagation of n-qubit system with transformed Hamiltonian
        tau = self.tau
        N = self.N
        
        #_, Q_dagger_nm1, J_subspace_nm1, _, _ = self.solve_for_Adjoint_Q_CG_method(n_qb-1)
        Q_dagger_Hz_Q = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        Q_dagger_Hx_Q = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        Q_dagger_Hy_Q = np.zeros((2**n_qb, 2**n_qb), dtype=np.complex128)
        Q_list = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
        Q_dagger_list = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
        for i in range(0, n_qb):
            Q_dagger_switch, Q_switch = self.index_switch(n_qb, i, int(n_qb-1) )
            #Q_dagger_dp, Q_dp = self.direct_product_single_qubit_S_n_minus_1(n_qb, Q_dagger_nm1, J_subspace_nm1)
            Q_dagger_list[i] = copy.deepcopy(Q_dagger_switch)
            Q_list[i] = copy.deepcopy(Q_switch)
            
        Q_dagger_Hz_Q = copy.deepcopy(Hz[n_qb-1])
        Q_dagger_Hx_Q = copy.deepcopy(Hx[n_qb-1])
        Q_dagger_Hy_Q = copy.deepcopy(Hy[n_qb-1])
               
        Q_dagger_H0_Q = 0.5 * omega * Q_dagger_Hz_Q
        H_cpl_sum = 0.25 * coupling * np.sum(H_cpl, axis=0)
        unitary_cpl_by_2_by_n_qb = expm(- 1j * (tau / (2 * n_qb) ) * H_cpl_sum )
        unitary_cpl_by_n_qb = expm(- 1j * (tau / n_qb) * H_cpl_sum )
        
        #dim_subspace_list = np.int64(2 * (2 * J_subspace_nm1 + 1) )  # dimension of subspace in H(2^n, C)
        
        Q_dagger_H0_Q_block = np.zeros((2, 2), dtype=np.float64)
        Q_dagger_Hx_Q_block = np.zeros((2, 2), dtype=np.float64)
        Q_dagger_Hy_Q_block = np.zeros((2, 2), dtype=np.complex128)
        mesh_seq = np.arange(0, 2, 1)
        block_idx = np.ix_(mesh_seq, mesh_seq)
        Q_dagger_H0_Q_block = Q_dagger_H0_Q[block_idx]
        Q_dagger_Hx_Q_block = Q_dagger_Hx_Q[block_idx]
        Q_dagger_Hy_Q_block = Q_dagger_Hy_Q[block_idx]
        
        unitary_list = np.zeros(( int(((N-1)/100) + 1), 2**n_qb, 2**n_qb), dtype=np.complex128)
        unitary_list[0] = copy.deepcopy(unitary)
        
        # solve for unitary
        unitary = np.matmul(unitary_cpl_by_2_by_n_qb, unitary)
        
        for j in range(0, N-1):
            propagator_U = np.identity((2**n_qb), dtype=np.complex128) 
            for i in range(0, n_qb):
                propagator_U_i = np.zeros((2**n_qb, 2**n_qb), dtype=np.complex128)
                idx = 0
                Hamiltonian_i_k = Q_dagger_H0_Q_block + 0.5 * Bx[i, j] * Q_dagger_Hx_Q_block + 0.5 * By[i, j] * Q_dagger_Hy_Q_block
                propagator_U_i_k = expm(- 1j * tau * Hamiltonian_i_k )   # to be block diagonalized
                for k in range(0, int(2**(n_qb - 1)) ):
                    mesh_seq = np.arange(idx, (idx + 2 ), 1)
                    block_idx = np.ix_(mesh_seq, mesh_seq)
                    propagator_U_i[block_idx] = copy.deepcopy(propagator_U_i_k )
                    idx = idx + 2
                propagator_U_i = np.matmul(np.matmul(Q_list[i], propagator_U_i), Q_dagger_list[i])
                propagator_U = np.matmul(propagator_U_i, propagator_U)
                propagator_U = np.matmul(unitary_cpl_by_n_qb, propagator_U)   # cpl terms
            unitary = np.matmul(propagator_U, unitary)
            if ((j+1) % 100) == 0 :
                unitary_list[int((j+1) / 100)] = copy.deepcopy(unitary)
                
        unitary = np.matmul(np.conjugate(np.transpose(unitary_cpl_by_2_by_n_qb) ), unitary)
        
        return unitary, unitary_list
        
    def compare_two_propagation(self, n_qb):
    # for 3 qubits and up
        N = self.N
        idn = self.idn
        sigmaz = self.sigmaz
        sigmax = self.sigmax
        sigmay = self.sigmay
        
        omega = 2.0
        coupling = 0.8
        random_seed_1 = 42
        random_seed_2 = 5
        np.random.seed(random_seed_1)   # remove the number when not testing for timing
        Bx = np.random.uniform(-1.0, 1.0, (n_qb, N-1) )
        np.random.seed(random_seed_2)   # remove the number when not testing for timing
        By = np.random.uniform(-1.0, 1.0, (n_qb, N-1) )
        
        flag = np.identity(n_qb, dtype=np.int8)
        Hz = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
        for j in range(0, n_qb):
            flag_j = flag[j]
            element_for_H_j = np.zeros((n_qb, 2, 2), dtype=np.float64)
            for k in range(0, n_qb):
                if flag_j[k] == 0:
                    element_for_H_j[k] = idn
                else:
                    element_for_H_j[k] = sigmaz
            H_j = np.kron(element_for_H_j[0], element_for_H_j[1])
            for k in range(2, n_qb):
                H_j = np.kron(H_j, element_for_H_j[k])
            Hz[j] = H_j
        
        Hx = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
        for j in range(0, n_qb):
            flag_j = flag[j]
            element_for_H_j = np.zeros((n_qb, 2, 2), dtype=np.float64)
            for k in range(0, n_qb):
                if flag_j[k] == 0:
                    element_for_H_j[k] = idn
                else:
                    element_for_H_j[k] = sigmax
            H_j = np.kron(element_for_H_j[0], element_for_H_j[1])
            for k in range(2, n_qb):
                H_j = np.kron(H_j, element_for_H_j[k])
            Hx[j] = H_j
        
        Hy = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.complex128)
        for j in range(0, n_qb):
            flag_j = flag[j]
            element_for_H_j = np.zeros((n_qb, 2, 2), dtype=np.complex128)
            for k in range(0, n_qb):
                if flag_j[k] == 0:
                    element_for_H_j[k] = idn
                else:
                    element_for_H_j[k] = sigmay
            H_j = np.kron(element_for_H_j[0], element_for_H_j[1])
            for k in range(2, n_qb):
                H_j = np.kron(H_j, element_for_H_j[k])
            Hy[j] = H_j
        
        flag = np.identity(n_qb, dtype=np.int8)    #whether sigmaz or identity-2
        for j in range(0, n_qb-1):
            flag[j][j+1] = 1             # 1 for sigmaz, 0 for identity-2
        flag[n_qb-1, 0] = 1
        Hz_coupling = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
        for j in range(0, n_qb):
            flag_j = flag[j]
            element_for_H_coupling_j = np.zeros((n_qb, 2, 2), dtype=np.float64)
            for k in range(0, n_qb):
                if flag_j[k] == 0:
                    element_for_H_coupling_j[k] = idn
                else:
                    element_for_H_coupling_j[k] = sigmaz
            H_coupling_j = np.kron(element_for_H_coupling_j[0], element_for_H_coupling_j[1])
            for k in range(2, n_qb):
                H_coupling_j = np.kron(H_coupling_j, element_for_H_coupling_j[k])
            Hz_coupling[j] = H_coupling_j
        
        unitary = np.identity((2**n_qb), dtype=np.complex128)
        
        t_start = time.time()
        unitary_original, unitary_original_list = self.fwd_propagation_original(n_qb, omega, Bx, By, Hz, Hx, Hy, unitary)
        t_nocpl_original = time.time() - t_start
        
        t_start = time.time()
        unitary_transformed, unitary_transformed_list = self.fwd_propagation_transformed(n_qb, omega, Bx, By, Hz, Hx, Hy, unitary)
        t_nocpl_transformed = time.time() - t_start
        
        t_start = time.time()
        unitary_cpl_original, unitary_cpl_original_list = self.fwd_propagation_cpl_original(n_qb, omega, coupling, Bx, By, Hz, Hx, Hy, Hz_coupling, unitary)
        t_cpl_original = time.time() - t_start
        
        t_start = time.time()
        unitary_cpl_transformed, unitary_cpl_transformed_list = self.fwd_propagation_cpl_transformed(n_qb, omega, coupling, Bx, By, Hz, Hx, Hy, Hz_coupling, unitary)
        t_cpl_transformed = time.time() - t_start
        
        fidelity = (np.abs(np.trace(np.matmul(np.conjugate(np.transpose(unitary_original) ), unitary_transformed) ) )**2 ) / ((2**n_qb)**2)
        
        fidelity_cpl = (np.abs(np.trace(np.matmul(np.conjugate(np.transpose(unitary_cpl_original) ), unitary_cpl_transformed) ) )**2 ) / ((2**n_qb)**2)
        
        fidelity_list = np.zeros((int(((N-1)/100) + 1) ), dtype=np.float64)
        fidelity_cpl_list = np.zeros((int(((N-1)/100) + 1) ), dtype=np.float64)
        for i in range(0, int(((N-1)/100) + 1) ):
            fidelity_i = (np.abs(np.trace(np.matmul(np.conjugate(np.transpose(unitary_original_list[i]) ), unitary_transformed_list[i]) ) )**2 ) / ((2**n_qb)**2)
            fidelity_cpl_i = (np.abs(np.trace(np.matmul(np.conjugate(np.transpose(unitary_cpl_original_list[i]) ), unitary_cpl_transformed_list[i]) ) )**2 ) / ((2**n_qb)**2)
            fidelity_list[i] = fidelity_i
            fidelity_cpl_list[i] = fidelity_cpl_i
        t_unitary = np.arange(0.0, 1000.0 + self.tau/2, self.tau * 100, dtype=np.float64)
        '''
        curr_name = 'QOC_LTS_fidelity_' + str(n_qb) + 'qb.mat'

        sio.savemat(curr_name, {'fidelity_list':np.array(fidelity_list, dtype=np.float64),
                                 'fidelity_cpl_list':np.array(fidelity_cpl_list, dtype=np.float64),
                                 't_unitary':t_unitary,
                                 't_nocpl_original':t_nocpl_original,
                                 't_nocpl_transformed':t_nocpl_transformed,
                                 't_cpl_original':t_cpl_original,
                                 't_cpl_transformed':t_cpl_transformed})
        
        '''
        return fidelity

    def index_switch(self, n_qb, idx1, idx2):
    # Switch the index of qubit idx1 and idx2 in a n-qubit system
    # It must be satisfied that n_qb > idx1, idx2 >= 0
        spin_up_down_list = np.zeros((2**n_qb, n_qb), dtype=np.int8)
        # listing the spin (up or down) of each qubit for the 2**n_qb basis, 0 for spin-up, 1 for spin-down
        for i in range(0, 2**n_qb):
            idx = copy.deepcopy(i)
            for k in range(0, n_qb):
                if idx < 2 ** (int(n_qb - 1 - k) ):
                    spin_up_down_list[i][k] = 0    # 0 for spin-up, 1 for spin-down
                else:
                    spin_up_down_list[i][k] = 1
                    idx = idx - 2 ** (int(n_qb - 1 - k) )
        #print(spin_up_down_list)
        
        idx_list = np.zeros((2**n_qb), dtype=np.int64)
        for i in range(0, 2**n_qb):
            spin_up_down_list_i = copy.deepcopy(spin_up_down_list[i] )
            spin_up_down_list_i_idx1 = copy.deepcopy(spin_up_down_list_i[idx1] )
            spin_up_down_list_i[idx1] = copy.deepcopy(spin_up_down_list_i[idx2] )
            spin_up_down_list_i[idx2] = copy.deepcopy(spin_up_down_list_i_idx1 )
            for k in range(0, 2**n_qb):
                if np.array_equal(spin_up_down_list_i, spin_up_down_list[k]) :
                    idx_list[i] = k
                    break
        #print(idx_list)
        
        Q_dagger = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        for i in range(0, 2**n_qb):
            Q_dagger[i][idx_list[i] ] = 1.0
        Q = copy.deepcopy(Q_dagger)    # Here Q_dagger = Q
        
        return Q_dagger, Q
    
    def direct_product_single_qubit_S_n_minus_1(self, n_qb, Q_dagger_nm1, J_subspace_nm1):
    # Direct product of the single qubit space and the S_(n-1)-decomposed H(2^(n-1), C) Hilbert space 
        spinup = self.spinup
        spindown = self.spindown
        
        dim_subspace_list = np.int64(2 * J_subspace_nm1 + 1)  # dimension of subspace in H(2^(n-1), C)
        Q_dagger = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        Q_idx = 0
        Q_nm1_idx = 0
        for dim in dim_subspace_list:
            for i in range(0, dim):
                Q_dagger[Q_idx + i] = np.kron(spinup, Q_dagger_nm1[Q_nm1_idx + i])
            Q_idx = Q_idx + dim
            for i in range(0, dim):
                Q_dagger[Q_idx + i] = np.kron(spindown, Q_dagger_nm1[Q_nm1_idx + i])
            Q_idx = Q_idx + dim
            Q_nm1_idx = Q_nm1_idx + dim
        
        Q = np.transpose(Q_dagger)
        
        return Q_dagger, Q
    
    def direct_product_single_qubit_D_n_minus_1(self, n_qb, Q_dagger_nm1, dimension_subspace):
    # Direct product of the single qubit space and the D_(n-1)-decomposed H(2^(n-1), C) Hilbert space 
        spinup = self.spinup
        spindown = self.spindown
        
        Q_dagger = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        Q_idx = 0
        Q_nm1_idx = 0
        for dim in dimension_subspace:
            for i in range(0, dim):
                Q_dagger[Q_idx + i] = np.kron(spinup, Q_dagger_nm1[Q_nm1_idx + i])
            Q_idx = Q_idx + dim
            for i in range(0, dim):
                Q_dagger[Q_idx + i] = np.kron(spindown, Q_dagger_nm1[Q_nm1_idx + i])
            Q_idx = Q_idx + dim
            Q_nm1_idx = Q_nm1_idx + dim
        
        Q = np.transpose(Q_dagger)
        
        return Q_dagger, Q
    
    def direct_product_two_spaces(self, n_qb, Q_dagger_1, Q_dagger_2, dimension_subspace_1, dimension_subspace_2):
    # Direct product of two decomposed spaces       
        Q_dagger = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        Q_idx = 0
        Q_1_idx = 0
        Q_2_idx = 0
        for dim_1 in dimension_subspace_1:
            for dim_2 in dimension_subspace_2:
                for i in range(0, dim_1):
                    for k in range(0, dim_2):
                        Q_dagger[Q_idx + k] = np.kron(Q_dagger_1[Q_1_idx + i], Q_dagger_2[Q_2_idx + k])
                    Q_idx = Q_idx + dim_2
                Q_2_idx = Q_2_idx + dim_2
            Q_2_idx = 0
            Q_1_idx = Q_1_idx + dim_1
        
        Q = np.transpose(Q_dagger)
        
        return Q_dagger, Q
    
    def transition_prob(self, Bx, By, evalGrad):
    # calculate for the states and gradients with the exponential propagator
        tau = self.tau
        N = self.N
        alpha = self.alpha
        n_qb = self.n_qb
        n_subdiag = self.n_subdiag            # 2**(n_qb-1) if use original H
        '''
        H0_ub = self.H0_ub
        H0_lb = self.H0_lb
        Hx_ub = self.Hx_ub
        Hx_lb = self.Hx_lb
        Hy_ub = self.Hy_ub
        Hy_lb = self.Hy_lb
        '''
        idn_nqb = self.idn_nqb
        dim_H = self.dim_H
        
        init_state = self.init_state    
        target_state = self.target_state
        
        Q = self.Q
        Q_dagger = self.Q_dagger
        Q_dagger_H0_Q = self.Q_dagger_H0_Q
        Q_dagger_Hx_Q = self.Q_dagger_Hx_Q
        Q_dagger_Hy_Q = self.Q_dagger_Hy_Q
        #mat_U_diag = self.mat_U_diag
        #mat_U_diag = [mat_U_diag]   # make mat_U_diag 2-dimensional array
        
        all_state = np.zeros((N, dim_H), dtype=np.complex128)
        init_state = np.matmul(Q_dagger, init_state)
        target_state = np.matmul(Q_dagger, target_state)        
        all_state[0] = init_state
        previous_state = all_state[0]
        
        # solve for states in the forward propagation
        for j in range(0, N-1):
            propagator_U = expm(- 1j * tau * (Q_dagger_H0_Q + Bx[j] * Q_dagger_Hx_Q + By[j] * Q_dagger_Hy_Q) )
            previous_state = np.matmul(propagator_U, previous_state)
            all_state[j+1] = previous_state
        
        final_state = all_state[N-1]
        P = np.power(np.abs(np.matmul(np.conjugate(target_state), final_state) ), 2)   # calculate transition probability
        J = P # - alpha * (np.sum(Bx**2) + np.sum(By**2)) * tau                        # calculate objective function

        if not evalGrad:
            return J, P, final_state
        else:
            #all_state = np.transpose(np.matmul(Q, np.transpose(all_state) ) )
            
            gradx = np.zeros((N-1, dim_H), dtype=np.complex128)
            grady = np.zeros((N-1, dim_H), dtype=np.complex128)
            
            #begin calculation for gradx and grady for the j=N-2 case
            propagator_U_half = expm(- 1j * (tau/2) * (Q_dagger_H0_Q + Bx[N-2] * Q_dagger_Hx_Q + By[N-2] * Q_dagger_Hy_Q) )
            propagator_U_half_Hx = np.matmul(propagator_U_half, Q_dagger_Hx_Q)
            propagator_U_half_Hy = np.matmul(propagator_U_half, Q_dagger_Hy_Q)
            gradx[N-2] = - 1j * (tau/2) * np.matmul(propagator_U_half_Hx, (all_state[N-1] + all_state[N-2]) )
            grady[N-2] = - 1j * (tau/2) * np.matmul(propagator_U_half_Hy, (all_state[N-1] + all_state[N-2]) )
            #end calculation for gradx and grady for the j=N-2 case

            #begin calculation for grad_Ef for the j=N-3 case
            latter_propagator_U = expm(- 1j * tau * (Q_dagger_H0_Q + Bx[N-2] * Q_dagger_Hx_Q + By[N-2] * Q_dagger_Hy_Q) )
            propagator_U_half = expm(- 1j * (tau/2) * (Q_dagger_H0_Q + Bx[N-3] * Q_dagger_Hx_Q + By[N-3] * Q_dagger_Hy_Q) )
            propagator_U_half_Hx = np.matmul(propagator_U_half, Q_dagger_Hx_Q)
            propagator_U_half_Hy = np.matmul(propagator_U_half, Q_dagger_Hy_Q)
            gradx[N-3] = - 1j * (tau/2) * np.matmul(latter_propagator_U, np.matmul(propagator_U_half_Hx, (all_state[N-2] + all_state[N-3]) ) )
            grady[N-3] = - 1j * (tau/2) * np.matmul(latter_propagator_U, np.matmul(propagator_U_half_Hy, (all_state[N-2] + all_state[N-3]) ) )
            #end calculation for gradx and grady for the j=N-3 case
            
            #begin calculation for grad_Ef for the 0<=j<=N-4 case
            for j in range(N-4, -1, -1):
                latter_propagator_U = np.matmul(latter_propagator_U, expm(- 1j * tau * (Q_dagger_H0_Q + Bx[j+1] * Q_dagger_Hx_Q + By[j+1] * Q_dagger_Hy_Q) ) )
                propagator_U_half = expm(- 1j * (tau/2) * (Q_dagger_H0_Q + Bx[j] * Q_dagger_Hx_Q + By[j] * Q_dagger_Hy_Q) )
                propagator_U_half_Hx = np.matmul(propagator_U_half, Q_dagger_Hx_Q)
                propagator_U_half_Hy = np.matmul(propagator_U_half, Q_dagger_Hy_Q)
                gradx[j] = - 1j * (tau/2) * np.matmul(latter_propagator_U, np.matmul(propagator_U_half_Hx, (all_state[j+1] + all_state[j]) ) )
                grady[j] = - 1j * (tau/2) * np.matmul(latter_propagator_U, np.matmul(propagator_U_half_Hy, (all_state[j+1] + all_state[j]) ) )
            #end calculation for grad_Ef for the 0<=j<=N-4 case
            
            # heuristically amplified gradient:
            if (np.abs(np.matmul(np.conjugate(target_state), final_state) ) < 0.1):
                scalar = 0.1 / np.abs(np.matmul(np.conjugate(target_state), final_state) )
                if (n_qb > 3) and (P < 0.001):
                    scalar = scalar * (800 ** (n_qb - 3) )
                    # 800 for init Bf = 0.00025 * random, 0.5 * Hx, 0.5 * Hy
                    # 40 for init Bf = 0.0005 * random, 0.5 * Hx, 0.5 * Hy
                    # 300 for init Bf = 0.0005 * random, 1 * Hx, 1 * Hy
                    # 100 for init Bf = 0.001 * random
                grad_Bx = 2 * np.real(scalar * np.matmul(np.conjugate(target_state), final_state) * np.conjugate(np.matmul(np.conjugate(target_state), np.transpose(gradx) ) ) ) # - 2 * alpha * tau * Bx
                grad_By = 2 * np.real(scalar * np.matmul(np.conjugate(target_state), final_state) * np.conjugate(np.matmul(np.conjugate(target_state), np.transpose(grady) ) ) ) # - 2 * alpha * tau * By 
            else:
                grad_Bx = 2 * np.real(np.matmul(np.conjugate(target_state), final_state) * np.conjugate(np.matmul(np.conjugate(target_state), np.transpose(gradx) ) ) ) # - 2 * alpha * tau * Bx
                grad_By = 2 * np.real(np.matmul(np.conjugate(target_state), final_state) * np.conjugate(np.matmul(np.conjugate(target_state), np.transpose(grady) ) ) ) # - 2 * alpha * tau * By 

            return J, P, final_state, grad_Bx, grad_By

    def golden_section_search(self, Bx0, By0, f0, dx, dy, ifContinuedSolving, iterNum, big_step=False):
    # search for the update rate gamma wth golden-section search
        # whether to search for gamma with big step-size:
        if ifContinuedSolving == False:
            if iterNum == 0:
                big_step = True
            
        if big_step == True:
            gamma = 1.0
            d_gamma = 0.01
        else:
            gamma = 0.1
            d_gamma = 1e-4

        def phi(gamma):
            obj, _, _ = self.transition_prob(Bx0+gamma*dx, By0+gamma*dy, evalGrad = False)
            return -obj

        def dphi(gamma):
            return (phi(gamma+d_gamma)-phi(gamma-d_gamma)) / (2*d_gamma)
        
        # calculate for upper limit of gamma:
        fl = f0     # f is alias for phi, which is -obj
        fr = phi(gamma)
        if big_step == True:
            a = 0.3
            b = 1.8
            threshold = 1e16
        else:
            a = 0.1
            b = 1.4
            threshold = 50.0           
        while fr < fl:
            gamma = (gamma + a) * b
            fl = fr
            fr = phi(gamma)
            if gamma > threshold:
                warnings.warn("Bisection linesearch: step length too large")
                break
        
        # golden-section search:
        gamma_l = np.finfo(float).eps               #gamma_left
        gamma_r = gamma                             #gamma_right
        golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0   # 0.618
        gamma_ml = gamma_r - (gamma_r - gamma_l) * golden_ratio   #gamma_middle_left
        gamma_mr = gamma_l + (gamma_r - gamma_l) * golden_ratio   #gamma_middle_right
        phi_l = phi(gamma_l)
        phi_r = phi(gamma_r)
        phi_ml = phi(gamma_ml)
        phi_mr = phi(gamma_mr)
        
        # search for gamma till (gamma_r - gamma_l) < tolerance:
        if big_step == True:
            tolerance = 1.0
        else:
            tolerance = 1e-4           
        while abs(gamma_r - gamma_l) > tolerance:   #1e-2
            if (np.amin([phi_l, phi_ml]) < np.amin([phi_mr, phi_r]) ):
                gamma_r = gamma_mr
                gamma_mr = gamma_ml
                gamma_ml = gamma_r - (gamma_r - gamma_l) * golden_ratio
                phi_r = phi_mr
                phi_mr = phi_ml
                phi_ml = phi(gamma_ml)
                
            else:
                gamma_l = gamma_ml
                gamma_ml = gamma_mr
                gamma_mr = gamma_l + (gamma_r - gamma_l) * golden_ratio
                phi_l = phi_ml
                phi_ml = phi_mr
                phi_mr = phi(gamma_mr)
                
        gamma = (gamma_mr + gamma_ml) / 2.0
        
        return gamma

    #@profile             # the profile function outputs runtime in each function
    def optimize(self, Bx, By, fidelity, maxIter):
    # solve for optimized fields, output final state and fidelity
        iterNum = 0
        t1 = time.time()
        
        ifContinuedSolving = self.ifContinuedSolving
        # record amplitude of fields and gradients:
        if ifContinuedSolving == False:
            J = []
            P = []
            Bx_amp_list = []
            By_amp_list = []
            grad_Bx_amp_list = []
            grad_By_amp_list = []
            Bx_amp = np.sqrt(np.matmul(Bx, Bx) )
            By_amp = np.sqrt(np.matmul(By, By) )
            Bx_amp_list.append(Bx_amp)
            By_amp_list.append(By_amp)
        else:
            J = self.J
            P = self.P
            Bx_amp_list = self.Bx_amp_list
            By_amp_list = self.By_amp_list
            grad_Bx_amp_list = self.grad_Bx_amp_list
            grad_By_amp_list = self.grad_By_amp_list   
        
        # gradient-based optimization:
        while iterNum < maxIter:
            t_start = time.time()
            obj, prob, final_state, grad_Bx, grad_By = self.transition_prob(Bx, By, evalGrad = True)
            
            if ifContinuedSolving == False:
                J.append(obj)
                P.append(prob)
            else:
                J = np.append(J, [obj], axis=0)
                P = np.append(P, [prob], axis=0)
            ids = "*"*20 + " Iteration # " + str(iterNum) + " " + "*"*(21-len(str(iterNum)))
            print(ids)
            print("\t=> J\t\t=\t" + str(round(J[-1], 4)) +"\n\t=> P\t\t=\t" + str(round(P[-1], 4)))   # print J and P
            
            if prob >= fidelity:           
                break                                           
            
            gamma = self.golden_section_search(Bx, By, -obj, grad_Bx, grad_By, ifContinuedSolving, iterNum)            
            Bx += gamma * grad_Bx
            By += gamma * grad_By
            
            Bx_amp = np.sqrt(np.matmul(Bx, Bx) )
            By_amp = np.sqrt(np.matmul(By, By) )
            grad_Bx_amp = np.sqrt(np.matmul(grad_Bx, grad_Bx) )
            grad_By_amp = np.sqrt(np.matmul(grad_By, grad_By) )
            if ifContinuedSolving == False:
                Bx_amp_list.append(Bx_amp)
                By_amp_list.append(By_amp)
                grad_Bx_amp_list.append(grad_Bx_amp)
                grad_By_amp_list.append(grad_By_amp)
            else:
                Bx_amp_list = np.append(Bx_amp_list, [Bx_amp], axis=0)
                By_amp_list = np.append(By_amp_list, [By_amp], axis=0)
                grad_Bx_amp_list = np.append(grad_Bx_amp_list, [grad_Bx_amp], axis=0)
                grad_By_amp_list = np.append(grad_By_amp_list, [grad_By_amp], axis=0)
            
            print("\t=> Step-size\t=\t" + str(round(gamma, 4)))                             # print update rate gamma
            t_iter = time.time() - t_start
            print("\t=> Time taken\t=\t" + str(int(t_iter)//60) + "m " + str(int(t_iter)%60) + "s")   # print timing
            print("*"*len(ids) + "\n")            
            iterNum += 1
        
        tT = time.time() - t1
        # print overall results after the last iteration:
        ids = "*"*16 + " Optimization complete " + "*"*16
        print("\n" + ids)
        print("\t=> J\t\t=\t" + str(round(J[-1], 4)) + "\n\t=> P\t\t=\t" + str(round(P[-1], 4)) + "\n\t=> Time taken\t=\t" + str(int(tT)//60) + "m " + str(int(tT)%60) + "s")
        print("*"*len(ids) + "\n")
        
        return Bx, By, final_state, J, P, Bx_amp_list, By_amp_list, grad_Bx_amp_list, grad_By_amp_list
    
    def diag_operator_method_Dn(self, n_qb):
    # Output Q_dagger, Q, dimension_subspace, M_eigenstate to decompose the Hamiltonians based on Dn symmetry
    # for 3 qubits and above
        diag_unitary_irrep_operator, diag_unitary_irrep_operator_coefficient_dec, diag_unitary_irrep_operator_coefficient_period = self.diag_unitary_irrep_operator_Dn(n_qb)
        all_possible_basis_Dn, basis_split_number, idx_all_possible_basis_Dn, dimension_candidate_subspace, candidate_basis_for_diag_operator, shift_number_candidate_subspace, shift_parity_candidate_subspace = self.basis_for_diag_unitary_irrep_operator_Dn(n_qb, diag_unitary_irrep_operator)
        Q_full, Q_dagger_full, dimension_subspace, M_eigenstate = self.solve_for_Adjoint_Q_diag_method_Dn(n_qb, 
            diag_unitary_irrep_operator, diag_unitary_irrep_operator_coefficient_dec, diag_unitary_irrep_operator_coefficient_period, all_possible_basis_Dn, idx_all_possible_basis_Dn, dimension_candidate_subspace, candidate_basis_for_diag_operator, shift_number_candidate_subspace, shift_parity_candidate_subspace)
        
        # Verify the dimension of subspaces
        Sn_to_Dn_decomposition = self.character_table(n_qb)
        young_tableau, young_diagram, dim_irrep, _, _, _ = self.Young_tableau(n_qb)
        _, weyl_tableau_num, _ = self.Weyl_tableau(n_qb, young_tableau, young_diagram, dim_irrep)
        dimension_subspace_from_Sn_to_Dn_decomposition = np.matmul(weyl_tableau_num, Sn_to_Dn_decomposition)
        
        if n_qb % 2 == 0 :     # if n_qb is even
            for i in range(0, 4):
                if dimension_subspace_from_Sn_to_Dn_decomposition[i] != dimension_subspace[i] :
                    raise ArithmeticError('The dimension of subspace ' + str(i) + ' do not match!')
            idx = 4
            for i in range(4, len(dimension_subspace_from_Sn_to_Dn_decomposition) ):
                if dimension_subspace_from_Sn_to_Dn_decomposition[i] != dimension_subspace[idx] :
                    raise ArithmeticError('The dimension of subspace ' + str(idx) + ' do not match!')
                if dimension_subspace_from_Sn_to_Dn_decomposition[i] != dimension_subspace[idx+1] :
                    raise ArithmeticError('The dimension of subspace ' + str(idx+1) + ' do not match!')
                idx = idx + 2
        else:                  # if n_qb is odd
            for i in range(0, 2):
                if dimension_subspace_from_Sn_to_Dn_decomposition[i] != dimension_subspace[i] :
                    raise ArithmeticError('The dimension of subspace ' + str(i) + ' do not match!')
            idx = 2
            for i in range(2, len(dimension_subspace_from_Sn_to_Dn_decomposition) ):
                if dimension_subspace_from_Sn_to_Dn_decomposition[i] != dimension_subspace[idx] :
                    raise ArithmeticError('The dimension of subspace ' + str(idx) + ' do not match!')
                if dimension_subspace_from_Sn_to_Dn_decomposition[i] != dimension_subspace[idx+1] :
                    raise ArithmeticError('The dimension of subspace ' + str(idx+1) + ' do not match!')
                idx = idx + 2
        
        return Q_full, Q_dagger_full, dimension_subspace, M_eigenstate, basis_split_number
    
    def diag_unitary_irrep_operator_Dn(self, n_qb):
    # Output diag_unitary_irrep_operator and diag_unitary_irrep_operator_coefficient in decimal number.
        # Generate diag_unitary_irrep_operator and diag_unitary_irrep_operator_coefficient_dec:
        '''
        if n_qb == 2:
            diag_unitary_irrep_operator = np.array([ [1, 2], [2, 1] ], dtype=np.int64)
            diag_unitary_irrep_operator_coefficient_dec = np.array([ [0.5, 0.5], [0.5, -0.5] ], dtype=np.float64)
            diag_unitary_irrep_operator_coefficient_period = np.array([1, 1], dtype=np.int64)
        else:
        '''
        if (n_qb % 2) == 0:     #if n_qb is even
            diag_unitary_irrep_operator = np.zeros((2 * n_qb, n_qb), dtype=np.int64)
            diag_unitary_irrep_operator_coefficient_dec = np.zeros((n_qb + 2, 2 * n_qb), dtype=np.float64) # 4 one-dimensional irreps, (n_qb-2)/2 two-dimensional irreps
            diag_unitary_irrep_operator_coefficient_period = np.zeros((n_qb + 2), dtype=np.int64)
            for i in range(0, n_qb):
                diag_unitary_irrep_operator[0][i] = int(i+1)
            for i in range(1, n_qb):
                diag_unitary_irrep_operator[i] = np.concatenate((diag_unitary_irrep_operator[0][i:], diag_unitary_irrep_operator[0][:i] ), axis=None)
            for i in range(0, n_qb):
                diag_unitary_irrep_operator[n_qb+i] = np.flip(diag_unitary_irrep_operator[i])
            # The rotation axis of C2_(i) or C2_(i') is shifted by (2*Pi/n_qb)/2 each time.
            # The elements of C2_(i) and C2_(i') appear alternately.
            
            diag_unitary_irrep_operator_coefficient_dec[0] = np.array([1.0] * int(2 * n_qb), dtype=np.float64)
            diag_unitary_irrep_operator_coefficient_dec[1] = np.array([1.0] * n_qb + [- 1.0] * n_qb, dtype=np.float64)
            for i in range(0, 2 * n_qb):
                if (i % 2) == 0:     #if i is even
                    diag_unitary_irrep_operator_coefficient_dec[2][i] = 1.0
                else:                #if i is odd
                    diag_unitary_irrep_operator_coefficient_dec[2][i] = - 1.0
            for i in range(0, n_qb):
                if (i % 2) == 0:     #if i is even
                    diag_unitary_irrep_operator_coefficient_dec[3][i] = 1.0
                else:                #if i is odd
                    diag_unitary_irrep_operator_coefficient_dec[3][i] = - 1.0
            for i in range(n_qb, 2 * n_qb):
                if (i % 2) == 0:     #if i is even
                    diag_unitary_irrep_operator_coefficient_dec[3][i] = - 1.0
                else:                #if i is odd
                    diag_unitary_irrep_operator_coefficient_dec[3][i] = 1.0
            idx = 4
            for i in range(0, int((n_qb-2)/2) ):           #(n_qb-2)/2 two-dimensional irreps
                for j in range(0, n_qb):
                    if (int(n_qb % 4) == 0) and (int(j * (i+1)) % int(n_qb / 2) == int(n_qb / 4) ):
                        diag_unitary_irrep_operator_coefficient_dec[idx][j] = 0.0
                        diag_unitary_irrep_operator_coefficient_dec[idx+1][j] = 0.0
                    else:
                        diag_unitary_irrep_operator_coefficient_dec[idx][j] = np.cos(2.0 * pi * float(j * (i+1) / n_qb) )
                        diag_unitary_irrep_operator_coefficient_dec[idx+1][j] = np.cos(2.0 * pi * float(j * (i+1) / n_qb) )
                for j in range(0, n_qb):
                    if (int(n_qb % 4) == 0) and (int(j * (i+1)) % int(n_qb / 2) == int(n_qb / 4) ):
                        diag_unitary_irrep_operator_coefficient_dec[idx][n_qb+j] = 0.0
                        diag_unitary_irrep_operator_coefficient_dec[idx+1][n_qb+j] = 0.0
                    else:
                        diag_unitary_irrep_operator_coefficient_dec[idx][n_qb+j] = np.cos(2.0 * pi * float(j * (i+1) / n_qb) )
                        diag_unitary_irrep_operator_coefficient_dec[idx+1][n_qb+j] = - np.cos(2.0 * pi * float(j * (i+1) / n_qb) )
                idx = idx + 2
            '''
            # The coefficients of each operator in group algebra:
            diag_unitary_irrep_operator_coefficient_dec[0] = diag_unitary_irrep_operator_coefficient_dec[0] * (1.0 / (2.0 * float(n_qb) ) )
            diag_unitary_irrep_operator_coefficient_dec[1] = diag_unitary_irrep_operator_coefficient_dec[1] * (1.0 / (2.0 * float(n_qb) ) )
            for i in range(0, int(n_qb) ):
                diag_unitary_irrep_operator_coefficient_dec[i+2] = diag_unitary_irrep_operator_coefficient_dec[i+2] * (1.0 / float(n_qb) )
            '''
            
            diag_unitary_irrep_operator_coefficient_period[0] = 1
            diag_unitary_irrep_operator_coefficient_period[1] = 1
            diag_unitary_irrep_operator_coefficient_period[2] = 2
            diag_unitary_irrep_operator_coefficient_period[3] = 2
            idx = 4
            for i in range(0, int((n_qb-2)/2) ):           #(n_qb-2)/2 two-dimensional irreps
                cyclic_number = int(i + 1)
                while (cyclic_number % n_qb != 0):
                    cyclic_number = int(cyclic_number + (i + 1) )
                period_i = int(cyclic_number / (i + 1) )
                diag_unitary_irrep_operator_coefficient_period[idx] = period_i
                diag_unitary_irrep_operator_coefficient_period[idx+1] = period_i
                idx = idx + 2
            
        else:                   #if n_qb is odd
            diag_unitary_irrep_operator = np.zeros((2 * n_qb, n_qb), dtype=np.int64)
            diag_unitary_irrep_operator_coefficient_dec = np.zeros((n_qb + 1, 2 * n_qb), dtype=np.float64) # 2 one-dimensional irreps, (n_qb-1)/2 two-dimensional irreps
            diag_unitary_irrep_operator_coefficient_period = np.zeros((n_qb + 1), dtype=np.int64)
            for i in range(0, n_qb):
                diag_unitary_irrep_operator[0][i] = int(i+1)
            for i in range(1, n_qb):
                diag_unitary_irrep_operator[i] = np.concatenate((diag_unitary_irrep_operator[0][i:], diag_unitary_irrep_operator[0][:i] ), axis=None)
            for i in range(0, n_qb):
                diag_unitary_irrep_operator[n_qb+i] = np.flip(diag_unitary_irrep_operator[i])
            # The rotation axis of C2_(i) is shifted by (2*Pi/n_qb)/2 each time.
            
            diag_unitary_irrep_operator_coefficient_dec[0] = np.array([1.0] * int(2 * n_qb), dtype=np.float64)
            diag_unitary_irrep_operator_coefficient_dec[1] = np.array([1.0] * n_qb + [- 1.0] * n_qb, dtype=np.float64)
            idx = 2
            for i in range(0, int((n_qb-1)/2) ):           #(n_qb-1)/2 two-dimensional irreps
                for j in range(0, n_qb):
                    diag_unitary_irrep_operator_coefficient_dec[idx][j] = np.cos(2.0 * pi * float(j * (i+1) / n_qb) )
                    diag_unitary_irrep_operator_coefficient_dec[idx+1][j] = np.cos(2.0 * pi * float(j * (i+1) / n_qb) )
                for j in range(0, n_qb):
                    diag_unitary_irrep_operator_coefficient_dec[idx][n_qb+j] = np.cos(2.0 * pi * float(j * (i+1) / n_qb) )
                    diag_unitary_irrep_operator_coefficient_dec[idx+1][n_qb+j] = - np.cos(2.0 * pi * float(j * (i+1) / n_qb) )
                idx = idx + 2
            '''
            # The coefficients of each operator in group algebra:
            diag_unitary_irrep_operator_coefficient_dec[0] = diag_unitary_irrep_operator_coefficient_dec[0] * (1.0 / (2.0 * float(n_qb) ) )
            diag_unitary_irrep_operator_coefficient_dec[1] = diag_unitary_irrep_operator_coefficient_dec[1] * (1.0 / (2.0 * float(n_qb) ) )
            for i in range(0, int(n_qb-1) ):
                diag_unitary_irrep_operator_coefficient_dec[i+2] = diag_unitary_irrep_operator_coefficient_dec[i+2] * (1.0 / float(n_qb) )
            '''
            
            diag_unitary_irrep_operator_coefficient_period[0] = 1
            diag_unitary_irrep_operator_coefficient_period[1] = 1
            idx = 2
            for i in range(0, int((n_qb-1)/2) ):           #(n_qb-1)/2 two-dimensional irreps
                cyclic_number = int(i + 1)
                while (cyclic_number % n_qb != 0):
                    cyclic_number = int(cyclic_number + (i + 1) )
                period_i = int(cyclic_number / (i + 1) )
                diag_unitary_irrep_operator_coefficient_period[idx] = period_i
                diag_unitary_irrep_operator_coefficient_period[idx+1] = period_i
                idx = idx + 2
        
        return diag_unitary_irrep_operator, diag_unitary_irrep_operator_coefficient_dec, diag_unitary_irrep_operator_coefficient_period
    
    def basis_for_diag_unitary_irrep_operator_Dn(self, n_qb, diag_unitary_irrep_operator):
    # Output the basis to be operated by diag_unitary_irrep_operator_Dn.
        # Generate the one-hot basis (Fock basis), 0 for spin-up, 1 for spin-down:
        # All basis with some certain number of spin-up's and spin-down's are put together.
        # Copied from self.solve_for_Adjoint_Q_Young_method()
        all_possible_basis = [ [[0, 0]], [[0, 1], [1, 0]], [[1, 1]] ]
        if n_qb > 2:
            for i in range(3, int(n_qb+1) ):
                all_possible_basis_new = []
                all_possible_basis_new_0 = [ [0] * i ]
                all_possible_basis_new.append(all_possible_basis_new_0)
                for j in range(1, i):
                    all_possible_basis_new_j = []
                    for k in range(0, comb(i-1, j-1, exact=True) ):
                        all_possible_basis_new_j.append(all_possible_basis[j-1][k] + [1] )
                    for k in range(0, comb(i-1, j, exact=True) ):
                        all_possible_basis_new_j.append(all_possible_basis[j][k] + [0] )
                    all_possible_basis_new.append(all_possible_basis_new_j)
                all_possible_basis_new_n_qb = [ [1] * i ]
                all_possible_basis_new.append(all_possible_basis_new_n_qb)
                all_possible_basis = copy.deepcopy(all_possible_basis_new)
        
        # D_n elements don't alter the length of spin-up's (spin-down's) and the distance between adjacent spin-up's (spin-down's).
        all_possible_basis_Dn = []
        basis_split_number = []            # number of Dn orbits in an Sn orbit
        all_possible_basis_Dn_0 = [ (0, ) * n_qb ]
        all_possible_basis_Dn.append(all_possible_basis_Dn_0)
        basis_split_number.append(1)
        for i in range(1, n_qb):
            all_possible_basis_i = copy.deepcopy(all_possible_basis[i])
            basis_split_number_i = 0
            while len(all_possible_basis_i) > 0:
                all_possible_basis_Dn_new = set()
                Dn_base_i_0 = all_possible_basis_i[0]
                all_possible_basis_Dn_new.add(tuple(Dn_base_i_0) )
                for j in range(0, int(2 * n_qb) ):
                    Dn_base_i_new = [0] * n_qb
                    for k in range(0, n_qb):
                        Dn_base_i_new[ diag_unitary_irrep_operator[j][k]-1 ] = Dn_base_i_0[k]
                    all_possible_basis_Dn_new.add(tuple(Dn_base_i_new) )
                for j in range(len(all_possible_basis_i)-1, -1, -1):
                    if (tuple(all_possible_basis_i[j]) in all_possible_basis_Dn_new ):
                        del all_possible_basis_i[j]
                all_possible_basis_Dn.append(list(all_possible_basis_Dn_new) )
                basis_split_number_i = basis_split_number_i + 1
            basis_split_number.append(basis_split_number_i)
        all_possible_basis_Dn_n_qb = [ (1, ) * n_qb ]
        all_possible_basis_Dn.append(all_possible_basis_Dn_n_qb)
        basis_split_number.append(1)
        
        all_possible_basis_Dn_list = []
        for i in all_possible_basis_Dn:
            all_possible_basis_Dn_i = copy.deepcopy(i)
            all_possible_basis_Dn_list_i = []
            for j in all_possible_basis_Dn_i:
                all_possible_basis_Dn_list_i.append(list(j) )
            all_possible_basis_Dn_list.append(all_possible_basis_Dn_list_i)
        
        # Each base will be a one-hot array, idx_all_possible_basis list the index of 1 for each array:
        idx_all_possible_basis = []
        idx_all_possible_basis_0 = [0]
        idx_all_possible_basis.append(idx_all_possible_basis_0)
        for i in range(1, len(all_possible_basis_Dn_list) ):
            idx_all_possible_basis_i = []
            all_possible_basis_Dn_list_i = copy.deepcopy(all_possible_basis_Dn_list[i] )
            for j in range(0, len(all_possible_basis_Dn_list_i) ):
                all_possible_basis_Dn_list_i_j = copy.deepcopy(all_possible_basis_Dn_list_i[j] )
                idx = 0
                for k in range(0, n_qb):
                    if all_possible_basis_Dn_list_i_j[k] == 1:
                        idx = idx + int(2**int(n_qb - 1 - k) )
                idx_all_possible_basis_i.append(idx)
            idx_all_possible_basis.append(idx_all_possible_basis_i)
        #print(idx_all_possible_basis)
        
        dimension_candidate_subspace = []
        for i in idx_all_possible_basis:
            dimension_candidate_subspace.append(len(i) )
        
        all_possible_basis_Dn_sorted = []
        idx_all_possible_basis_sorted = []
        all_possible_basis_Dn_sorted_0 = copy.deepcopy(all_possible_basis_Dn_list[0] )
        all_possible_basis_Dn_sorted.append(all_possible_basis_Dn_sorted_0)
        idx_all_possible_basis_sorted_0 = [0]
        idx_all_possible_basis_sorted.append(idx_all_possible_basis_sorted_0)
        for i in range(1, len(dimension_candidate_subspace) ):
            idx_all_possible_basis_i = copy.deepcopy(idx_all_possible_basis[i] )
            idx_sorted_i = np.argsort(idx_all_possible_basis_i)
            all_possible_basis_Dn_list_i = copy.deepcopy(all_possible_basis_Dn_list[i] )
            all_possible_basis_Dn_sorted_i = []
            for j in range(0, dimension_candidate_subspace[i] ):
                all_possible_basis_Dn_sorted_i.append(all_possible_basis_Dn_list_i[idx_sorted_i[j] ] )
            all_possible_basis_Dn_sorted.append(all_possible_basis_Dn_sorted_i)
            idx_all_possible_basis_sorted_i = sorted(idx_all_possible_basis_i)
            idx_all_possible_basis_sorted.append(idx_all_possible_basis_sorted_i)
                        
        candidate_basis_for_diag_operator = []
        for i in range(0, len(dimension_candidate_subspace) ):
            candidate_basis_for_diag_operator.append(all_possible_basis_Dn_sorted[i][0])
        #candidate_basis_for_diag_operator = np.array(candidate_basis_for_diag_operator, dtype=np.int8)
        
        shift_number_candidate_subspace = []             # candidate subspace is alias of orbit
        for i in range(0, len(dimension_candidate_subspace) ):
            if dimension_candidate_subspace[i] == int(2 * n_qb) :
                shift_number_candidate_subspace_i = []      # no shift number for candidate subspace with int(2 * n_qb) basis
            else:
                shift_number_candidate_subspace_i = []
                all_possible_basis_Dn_sorted_i = copy.deepcopy(all_possible_basis_Dn_sorted[i] )
                for j in range(0, dimension_candidate_subspace[i] ):
                    base_i_j = copy.deepcopy(all_possible_basis_Dn_sorted_i[j] )
                    flip_base_i_j = base_i_j[::-1]
                    if base_i_j == flip_base_i_j :
                        shift_number_i_j = 0       
                    else:
                        shift_number_i_j = 0
                        while (base_i_j != flip_base_i_j) and (shift_number_i_j < n_qb) :
                            base_i_j = [base_i_j[-1] ] + base_i_j[:-1]
                            shift_number_i_j = shift_number_i_j + 1
                    if shift_number_i_j < n_qb :
                        shift_number_candidate_subspace_i.append(shift_number_i_j)
            shift_number_candidate_subspace.append(shift_number_candidate_subspace_i)
        
        # Shift parity is not a good quantum number when n_qb is odd
        # And we do not need shift parity for D_n/2_(i) operators when n_qb is odd
        shift_parity_candidate_subspace = []
        for i in range(0, len(candidate_basis_for_diag_operator) ):
            base_i = candidate_basis_for_diag_operator[i]
            flip_base_i = base_i[::-1]
            if dimension_candidate_subspace[i] == int(2 * n_qb) :
                shift_parity_candidate_subspace_i = 2       # 2 for candidate subspace in which the states won't shift to the flipped states
            elif base_i == flip_base_i :
                shift_parity_candidate_subspace_i = 0       # 0 for even parity
            else:
                shift_number_i = 0
                while (base_i != flip_base_i) and (shift_number_i < n_qb) :
                    base_i = [base_i[-1] ] + base_i[:-1]
                    shift_number_i = shift_number_i + 1
                if shift_number_i == n_qb :
                    shift_parity_candidate_subspace_i = 2
                elif (shift_number_i % 2) == 0:
                    shift_parity_candidate_subspace_i = 0
                else:
                    shift_parity_candidate_subspace_i = 1    # 1 for odd parity
            shift_parity_candidate_subspace.append(shift_parity_candidate_subspace_i)
        
        return all_possible_basis_Dn_sorted, basis_split_number, idx_all_possible_basis_sorted, dimension_candidate_subspace, candidate_basis_for_diag_operator, shift_number_candidate_subspace, shift_parity_candidate_subspace
    
    def solve_for_Adjoint_Q_diag_method_Dn(self, n_qb, diag_unitary_irrep_operator, diag_unitary_irrep_operator_coefficient_dec, diag_unitary_irrep_operator_coefficient_period, all_possible_basis_Dn, idx_all_possible_basis_Dn, dimension_candidate_subspace, candidate_basis_for_diag_operator, shift_number_candidate_subspace, shift_parity_candidate_subspace):
    # Output the adjoint matrix Q and Q_dagger under Dn symmetry. Output dimension of subspace and M for each eigenstate.
    # M is a good quantum number under Dn symmetry.
        # The diag operators will operate on basis_for_each_diag_operator:
        basis_for_each_diag_operator = []
        idx_of_subspace_for_each_diag_operator = []
        to_be_orthogonalized_idx_list = []   # for Gram-Schmidt process of eigenstates in large candidate subspaces (orbits)
        if (n_qb % 2) == 0:     #if n_qb is even
        # values in diag_unitary_irrep_operator_coefficient_period can be even or odd
            basis_for_each_diag_operator_0 = copy.deepcopy(candidate_basis_for_diag_operator)   # basis for Id
            basis_for_each_diag_operator.append(basis_for_each_diag_operator_0)
            idx_of_subspace_for_each_diag_operator_0 = []
            for i in range(0, len(dimension_candidate_subspace) ):
                idx_of_subspace_for_each_diag_operator_0.append(i)
            idx_of_subspace_for_each_diag_operator.append(idx_of_subspace_for_each_diag_operator_0)
            to_be_orthogonalized_idx = len(idx_of_subspace_for_each_diag_operator[0])
            
            basis_for_each_diag_operator_1 = []   # basis for C
            idx_of_subspace_for_each_diag_operator_1 = []
            for i in range(0, len(dimension_candidate_subspace) ):
                if shift_parity_candidate_subspace[i] == 2 :
                    basis_for_each_diag_operator_1.append(candidate_basis_for_diag_operator[i])
                    idx_of_subspace_for_each_diag_operator_1.append(i)
            basis_for_each_diag_operator.append(basis_for_each_diag_operator_1)
            idx_of_subspace_for_each_diag_operator.append(idx_of_subspace_for_each_diag_operator_1)
            to_be_orthogonalized_idx = to_be_orthogonalized_idx + len(idx_of_subspace_for_each_diag_operator[1])
            
            basis_for_each_diag_operator_2 = []   # basis for D_n/2_(1), even parity
            idx_of_subspace_for_each_diag_operator_2 = []
            for i in range(1, len(dimension_candidate_subspace)-1 ):
                if dimension_candidate_subspace[i] % 2 == 0 :
                    if shift_parity_candidate_subspace[i] == 0 :
                        basis_for_each_diag_operator_2.append(candidate_basis_for_diag_operator[i])
                        idx_of_subspace_for_each_diag_operator_2.append(i)
                    elif shift_parity_candidate_subspace[i] == 2 :
                        if (dimension_candidate_subspace[i] / 2) % 2 == 0 :
                            basis_for_each_diag_operator_2.append(candidate_basis_for_diag_operator[i])
                            idx_of_subspace_for_each_diag_operator_2.append(i)
            basis_for_each_diag_operator.append(basis_for_each_diag_operator_2)
            idx_of_subspace_for_each_diag_operator.append(idx_of_subspace_for_each_diag_operator_2)
            to_be_orthogonalized_idx = to_be_orthogonalized_idx + len(idx_of_subspace_for_each_diag_operator[2])
            
            basis_for_each_diag_operator_3 = []   # basis for D_n/2_(2), odd parity
            idx_of_subspace_for_each_diag_operator_3 = []
            for i in range(1, len(dimension_candidate_subspace)-1 ):
                if dimension_candidate_subspace[i] % 2 == 0 :
                    if shift_parity_candidate_subspace[i] == 1 :
                        basis_for_each_diag_operator_3.append(candidate_basis_for_diag_operator[i])
                        idx_of_subspace_for_each_diag_operator_3.append(i)
                    elif shift_parity_candidate_subspace[i] == 2 :
                        if (dimension_candidate_subspace[i] / 2) % 2 == 0 :
                            basis_for_each_diag_operator_3.append(candidate_basis_for_diag_operator[i])
                            idx_of_subspace_for_each_diag_operator_3.append(i)
            basis_for_each_diag_operator.append(basis_for_each_diag_operator_3)
            idx_of_subspace_for_each_diag_operator.append(idx_of_subspace_for_each_diag_operator_3)
            to_be_orthogonalized_idx = to_be_orthogonalized_idx + len(idx_of_subspace_for_each_diag_operator[3])
            
            for i in range(4, len(diag_unitary_irrep_operator_coefficient_period) ):
                basis_for_each_diag_operator_i = []   # basis for 2-dimensional irreps
                idx_of_subspace_for_each_diag_operator_i = []
                # To make the two basis as independent as possible, reduce float error in Gram-Schmidt process:
                abs_diag_unitary_irrep_operator_coefficient_dec_i = np.abs(diag_unitary_irrep_operator_coefficient_dec[i][:diag_unitary_irrep_operator_coefficient_period[i] ] )
                shift_number_for_new_base_in_large_candidate_subspace = np.argmin(abs_diag_unitary_irrep_operator_coefficient_dec_i)
                if (i % 2 == 1) :
                    ifLowerElement = True
                else:
                    ifLowerElement = False
                for j in range(0, len(dimension_candidate_subspace) ):
                    valid_operation_flag = 0
                    if (shift_parity_candidate_subspace[j] < 2) and (dimension_candidate_subspace[j] % diag_unitary_irrep_operator_coefficient_period[i] == 0) :
                        valid_operation_flag = 1
                    elif (shift_parity_candidate_subspace[j] == 2) and ( (dimension_candidate_subspace[j] / 2) % diag_unitary_irrep_operator_coefficient_period[i] == 0) :
                        valid_operation_flag = 1
                    if valid_operation_flag == 1 :
                        if shift_parity_candidate_subspace[j] == 2 :
                            basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j])
                            idx_of_subspace_for_each_diag_operator_i.append(j)
                            basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j][-shift_number_for_new_base_in_large_candidate_subspace:] + candidate_basis_for_diag_operator[j][:-shift_number_for_new_base_in_large_candidate_subspace])
                            idx_of_subspace_for_each_diag_operator_i.append(j)
                            to_be_orthogonalized_idx_list.append(to_be_orthogonalized_idx)
                            to_be_orthogonalized_idx = to_be_orthogonalized_idx + 2
                        else:
                            if ifLowerElement:
                                if (shift_number_candidate_subspace[j][0] % diag_unitary_irrep_operator_coefficient_period[i] == 0) :
                                    idx = 0
                                    while (shift_number_candidate_subspace[j][idx] % diag_unitary_irrep_operator_coefficient_period[i] == 0) :
                                        idx = idx + 1
                                    candidate_basis_for_diag_operator_i_j = copy.deepcopy(all_possible_basis_Dn[j][idx] )
                                    basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator_i_j)
                                else:
                                    basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j])
                            else:   # if upper element
                                if (diag_unitary_irrep_operator_coefficient_period[i] % 2 == 1) :
                                    basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j])
                                else:        
                                    if (shift_number_candidate_subspace[j][0] % diag_unitary_irrep_operator_coefficient_period[i] == int(diag_unitary_irrep_operator_coefficient_period[i] / 2) ) :
                                        idx = 0
                                        while (shift_number_candidate_subspace[j][idx] % diag_unitary_irrep_operator_coefficient_period[i] == int(diag_unitary_irrep_operator_coefficient_period[i] / 2) ) :
                                            idx = idx + 1
                                        candidate_basis_for_diag_operator_i_j = copy.deepcopy(all_possible_basis_Dn[j][idx] )
                                        basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator_i_j)
                                    else:
                                        basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j])
                            idx_of_subspace_for_each_diag_operator_i.append(j)
                            to_be_orthogonalized_idx = to_be_orthogonalized_idx + 1
                basis_for_each_diag_operator.append(basis_for_each_diag_operator_i)
                idx_of_subspace_for_each_diag_operator.append(idx_of_subspace_for_each_diag_operator_i)
                
        else:                   #if n_qb is odd
        # values in diag_unitary_irrep_operator_coefficient_period must be odd
            basis_for_each_diag_operator_0 = copy.deepcopy(candidate_basis_for_diag_operator)   # basis for Id
            basis_for_each_diag_operator.append(basis_for_each_diag_operator_0)
            idx_of_subspace_for_each_diag_operator_0 = []
            for i in range(0, len(dimension_candidate_subspace) ):
                idx_of_subspace_for_each_diag_operator_0.append(i)
            idx_of_subspace_for_each_diag_operator.append(idx_of_subspace_for_each_diag_operator_0)
            to_be_orthogonalized_idx = len(idx_of_subspace_for_each_diag_operator[0])
            
            basis_for_each_diag_operator_1 = []   # basis for C
            idx_of_subspace_for_each_diag_operator_1 = []
            for i in range(0, len(dimension_candidate_subspace) ):
                if shift_parity_candidate_subspace[i] == 2 :
                    basis_for_each_diag_operator_1.append(candidate_basis_for_diag_operator[i])
                    idx_of_subspace_for_each_diag_operator_1.append(i)
            basis_for_each_diag_operator.append(basis_for_each_diag_operator_1)
            idx_of_subspace_for_each_diag_operator.append(idx_of_subspace_for_each_diag_operator_1)
            to_be_orthogonalized_idx = to_be_orthogonalized_idx + len(idx_of_subspace_for_each_diag_operator[1])
            
            for i in range(2, len(diag_unitary_irrep_operator_coefficient_period) ):
                basis_for_each_diag_operator_i = []   # basis for E
                idx_of_subspace_for_each_diag_operator_i = []
                # To make the two basis as independent as possible, reduce float error in Gram-Schmidt process:
                abs_diag_unitary_irrep_operator_coefficient_dec_i = np.abs(diag_unitary_irrep_operator_coefficient_dec[i][:diag_unitary_irrep_operator_coefficient_period[i] ] )
                shift_number_for_new_base_in_large_candidate_subspace = np.argmin(abs_diag_unitary_irrep_operator_coefficient_dec_i)
                if (i % 2 == 1) :
                    ifLowerElement = True
                else:
                    ifLowerElement = False
                for j in range(0, len(dimension_candidate_subspace) ):
                    valid_operation_flag = 0
                    if (shift_parity_candidate_subspace[j] < 2) and (dimension_candidate_subspace[j] % diag_unitary_irrep_operator_coefficient_period[i] == 0) :
                        valid_operation_flag = 1
                    elif (shift_parity_candidate_subspace[j] == 2) and ( (dimension_candidate_subspace[j] / 2) % diag_unitary_irrep_operator_coefficient_period[i] == 0) :
                        valid_operation_flag = 1
                    if valid_operation_flag == 1 :
                        if shift_parity_candidate_subspace[j] == 2 :
                            basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j])
                            idx_of_subspace_for_each_diag_operator_i.append(j)
                            basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j][-shift_number_for_new_base_in_large_candidate_subspace:] + candidate_basis_for_diag_operator[j][:-shift_number_for_new_base_in_large_candidate_subspace])
                            idx_of_subspace_for_each_diag_operator_i.append(j)
                            to_be_orthogonalized_idx_list.append(to_be_orthogonalized_idx)
                            to_be_orthogonalized_idx = to_be_orthogonalized_idx + 2
                        else:
                            if ifLowerElement:
                                if (shift_number_candidate_subspace[j][0] % diag_unitary_irrep_operator_coefficient_period[i] == 0) :
                                    idx = 0
                                    while (shift_number_candidate_subspace[j][idx] % diag_unitary_irrep_operator_coefficient_period[i] == 0) :
                                        idx = idx + 1
                                    candidate_basis_for_diag_operator_i_j = copy.deepcopy(all_possible_basis_Dn[j][idx] )
                                    basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator_i_j)
                                else:
                                    basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j])
                            else:
                                basis_for_each_diag_operator_i.append(candidate_basis_for_diag_operator[j])
                            idx_of_subspace_for_each_diag_operator_i.append(j)
                            to_be_orthogonalized_idx = to_be_orthogonalized_idx + 1
                basis_for_each_diag_operator.append(basis_for_each_diag_operator_i)
                idx_of_subspace_for_each_diag_operator.append(idx_of_subspace_for_each_diag_operator_i)
        
        base_coefficient_dec = []
        # Generate the principle (first) subspace (Id operator) :
        base_coefficient_dec_0 = []
        for i in range(0, len(dimension_candidate_subspace) ):
            base_coefficient_dec_0_i = [float(2 * n_qb / dimension_candidate_subspace[i])] * dimension_candidate_subspace[i]
            base_coefficient_dec_0.append(base_coefficient_dec_0_i)
        base_coefficient_dec.append(base_coefficient_dec_0)
        
        # Generate other subspaces:
        for i in range(1, len(diag_unitary_irrep_operator_coefficient_period) ):
            diag_unitary_irrep_operator_coefficient_dec_i = copy.deepcopy(diag_unitary_irrep_operator_coefficient_dec[i] )
            basis_for_diag_operator_i = copy.deepcopy(basis_for_each_diag_operator[i] )
            idx_of_subspace_for_diag_operator_i = copy.deepcopy(idx_of_subspace_for_each_diag_operator[i] )
            base_coefficient_dec_i = []
            for j in range(0, len(idx_of_subspace_for_diag_operator_i) ):
                basis_for_diag_operator_i_j = copy.deepcopy(basis_for_diag_operator_i[j] )
                base_coefficient_dec_i_j = [0.0] * dimension_candidate_subspace[idx_of_subspace_for_diag_operator_i[j] ]
                for k in range(0, int(2*n_qb) ):
                    basis_for_diag_operator_new = [0] * n_qb
                    for l in range(0, n_qb):
                        basis_for_diag_operator_new[ int(diag_unitary_irrep_operator[k][l] - 1) ] = basis_for_diag_operator_i_j[l]
                    for l in range(0, dimension_candidate_subspace[idx_of_subspace_for_diag_operator_i[j] ] ):
                        if all_possible_basis_Dn[idx_of_subspace_for_diag_operator_i[j] ][l] == basis_for_diag_operator_new:
                            base_coefficient_dec_i_j[l] = base_coefficient_dec_i_j[l] + diag_unitary_irrep_operator_coefficient_dec_i[k]
                base_coefficient_dec_i.append(base_coefficient_dec_i_j)
            base_coefficient_dec.append(base_coefficient_dec_i)
        
        # Generate Q and Q_dagger:
        Q_dagger = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        idx = 0
        for i in range(0, len(diag_unitary_irrep_operator_coefficient_period) ):
            idx_of_subspace_for_each_diag_operator_i = copy.deepcopy(idx_of_subspace_for_each_diag_operator[i] )
            base_coefficient_dec_i = copy.deepcopy(base_coefficient_dec[i] )
            for j in range(0, len(idx_of_subspace_for_each_diag_operator_i) ):
                base_coefficient_dec_i_j = copy.deepcopy(base_coefficient_dec_i[j] )
                idx_all_possible_basis_Dn_i_j = idx_all_possible_basis_Dn[idx_of_subspace_for_each_diag_operator_i[j] ]
                for k in range(0, len(idx_all_possible_basis_Dn_i_j) ):
                    Q_dagger[idx][idx_all_possible_basis_Dn_i_j[k] ] = base_coefficient_dec_i_j[k]          
                idx = idx + 1
        
        # Gram-Schmidt process for eigenstates in shift_parity = 2 candidate susbpaces:
        for i in to_be_orthogonalized_idx_list:
            Q_dagger[i+1] = Q_dagger[i+1] - (Q_dagger[i] * ( np.matmul(Q_dagger[i], Q_dagger[i+1]) / np.sum(Q_dagger[i]**2) ) )
        
        for i in range(0, 2**n_qb):
            Q_dagger[i] = Q_dagger[i] / np.sqrt(np.sum(Q_dagger[i]**2) )
            
        Q = np.transpose(Q_dagger)
        
        dimension_subspace = []
        for i in range(0, len(diag_unitary_irrep_operator_coefficient_period) ):
            dimension_subspace.append(len(idx_of_subspace_for_each_diag_operator[i]) )
        
        M_eigenstate = []
        for i in range(0, len(diag_unitary_irrep_operator_coefficient_period) ):
            M_eigenstate_i = []
            for j in range(0, len(idx_of_subspace_for_each_diag_operator[i]) ):
                M_eigenstate_i_j = 0.0
                for k in range(0, n_qb):
                    if basis_for_each_diag_operator[i][j][k] == 0 :
                        M_eigenstate_i_j = M_eigenstate_i_j + 0.5
                    else:
                        M_eigenstate_i_j = M_eigenstate_i_j - 0.5
                M_eigenstate_i.append(M_eigenstate_i_j)
            M_eigenstate.append(M_eigenstate_i)
        
        return Q, Q_dagger, dimension_subspace, M_eigenstate
    
    def character_table(self, n_qb):
    # Decompose the character table of Sn to Dn.
        young_diagram_irrep_Sn, young_diagram_class_Sn, elements_class_Sn, num_elements_class_Sn, character_table_Sn = self.character_table_Sn(n_qb)
        idx_irrep_Dn, elements_class_Dn, num_elements_class_Dn, character_table_Dn = self.character_table_Dn(n_qb)
        
        extended_elements_class_Sn = elements_class_Dn
        extended_num_elements_class_Sn = num_elements_class_Dn
        extended_character_table_Sn = np.zeros((len(young_diagram_irrep_Sn), len(extended_num_elements_class_Sn) ), dtype=np.float64)
        if (n_qb % 2) == 0:     #if n_qb is even
            for i in range(0, len(young_diagram_irrep_Sn) ):
                for j in range(0, len(extended_num_elements_class_Sn)-2 ):
                    for k in range(0, len(num_elements_class_Sn)-1 ):
                        if (extended_elements_class_Sn[j][0] in elements_class_Sn[k] ):
                            extended_character_table_Sn[i][j] = float(character_table_Sn[i][k] )
            for i in range(0, len(young_diagram_irrep_Sn) ):
                extended_character_table_Sn[i][-1] = float(character_table_Sn[i][-1] )
                extended_character_table_Sn[i][-2] = float(character_table_Sn[i][-2] )
        else:                   #if n_qb is odd
            for i in range(0, len(young_diagram_irrep_Sn) ):
                for j in range(0, len(extended_num_elements_class_Sn)-1 ):
                    for k in range(0, len(num_elements_class_Sn)-1 ):
                        if (extended_elements_class_Sn[j][0] in elements_class_Sn[k] ):
                            extended_character_table_Sn[i][j] = float(character_table_Sn[i][k] )
            for i in range(0, len(young_diagram_irrep_Sn) ):
                extended_character_table_Sn[i][-1] = float(character_table_Sn[i][-1] )
        
        if n_qb == 2:
            Sn_to_Dn_decomposition = np.array([ [1, 0], [0, 1] ], dtype=np.int64)
        else:
            Sn_to_Dn_decomposition = np.zeros((len(young_diagram_irrep_Sn), len(idx_irrep_Dn) ), dtype=np.int64)
            for i in range(0, len(young_diagram_irrep_Sn) ):
                for j in range(0, len(idx_irrep_Dn) ):
                    Sn_to_Dn_decomposition_i_j = (1.0 / (2.0 * n_qb) ) * np.sum(np.multiply(num_elements_class_Dn, np.multiply(np.conjugate(extended_character_table_Sn[i]), character_table_Dn[j] ) ) )
                    Sn_to_Dn_decomposition_i_j_round = round(Sn_to_Dn_decomposition_i_j)
                    if np.abs(Sn_to_Dn_decomposition_i_j_round - Sn_to_Dn_decomposition_i_j) < 0.001 :
                        Sn_to_Dn_decomposition[i][j] = Sn_to_Dn_decomposition_i_j_round
                    else:
                        raise ArithmeticError("Non-integer in Sn-to-Dn decomposition!")
        '''
        if n_qb == 2:
            print('[2 0] = Id * 1 + C * 0 + D_n/2 * 0')
            print('[1 1] = Id * 0 + C * 0 + D_n/2 * 1')
        else:
            if (n_qb % 2) == 0:     #if n_qb is even
                for i in range(0, len(young_diagram_irrep_Sn) ):
                    decomposition_i = ( str(young_diagram_irrep_Sn[i]) + ' = Id * ' + str(Sn_to_Dn_decomposition[i][0]) + ' + C * ' + str(Sn_to_Dn_decomposition[i][1]) 
                                       + ' + D_n/2_(1) * ' + str(Sn_to_Dn_decomposition[i][2]) + ' + D_n/2_(2) * ' + str(Sn_to_Dn_decomposition[i][3]) )
                    for j in range(4, len(idx_irrep_Dn) ):
                        decomposition_i = decomposition_i + ' + E' + str(j-3) + ' * ' + str(Sn_to_Dn_decomposition[i][j])
                    print(decomposition_i)
            else:                   #if n_qb is odd
                for i in range(0, len(young_diagram_irrep_Sn) ):
                    decomposition_i = ( str(young_diagram_irrep_Sn[i]) + ' = Id * ' + str(Sn_to_Dn_decomposition[i][0]) + ' + C * ' + str(Sn_to_Dn_decomposition[i][1]) )
                    for j in range(2, len(idx_irrep_Dn) ):
                        decomposition_i = decomposition_i + ' + E' + str(j-1) + ' * ' + str(Sn_to_Dn_decomposition[i][j])
                    print(decomposition_i)
        '''
        return Sn_to_Dn_decomposition
    
    def character_table_Sn(self, n_qb):
    # Output the character table of S_n for all elements in D_n and all irreps of one-row and two-row Young diagrams.
    # n is n_qb, n >= 2
        # Generate the Young diagrams of irreps, each irrep is denoted with a Young diagram:
        young_diagram_irrep = []
        if (n_qb % 2) == 0:     #if n_qb is even
            for i in range(0, int(n_qb/2)+1 ):
                young_diagram_irrep.append([int(n_qb-i), i])
        else:                   #if n_qb is odd
            for i in range(0, int((n_qb+1)/2 ) ):
                young_diagram_irrep.append([int(n_qb-i), i])
        young_diagram_irrep = np.array(young_diagram_irrep, dtype=np.int64)
        
        # Generate the Young diagrams of classes, each class is denoted with a Young diagram:
        # n_qb must >= 2
        young_diagram_class = []
        if (n_qb % 2) == 0:     #if n_qb is even
            # The classes are ordered to be e, Cn, Cn^2, ..., C2_i' and Cn^(n/2), C2_i
            young_diagram_class.append([1] * n_qb)
            for i in range(1, int(n_qb/2) ):
                if (n_qb % i) == 0:
                    young_diagram_class.append([int(n_qb/i)] * i )
            class_C2_i_prime = [2] * int(n_qb/2)
            young_diagram_class.append(class_C2_i_prime)
            if n_qb > 2:
                class_C2_i = [1] * 2 + [2] * int((n_qb/2)-1 )
                young_diagram_class.append(class_C2_i)            
        else:                   #if n_qb is odd
            # The classes are ordered to be e, Cn, ..., C2_i
            young_diagram_class.append([1] * n_qb)
            for i in range(1, int((n_qb+1)/2 ) ):
                if (n_qb % i) == 0:
                    young_diagram_class.append([int(n_qb/i)] * i )
            class_C2_i = [1] + [2] * int((n_qb-1)/2 )
            young_diagram_class.append(class_C2_i)
        
        # Generate the elements and number of elements in each class:
        # n_qb must >= 2
        elements_class = []
        num_elements_class = []
        if (n_qb % 2) == 0:     #if n_qb is even
            # The classes are ordered to be e, Cn, Cn^2, ..., (C2_i' and Cn^(n/2) ), C2_i
            # record the index i of Cn^i
            elements_class.append([0] )
            num_elements_class.append(1)
            for i in range(1, int(n_qb/2) ):
                if (n_qb % i) == 0:
                    elements_class_new = [i, int(n_qb - i)]
                    num_elements_class_new = 2
                    for j in range(2, n_qb):
                        if int(i * j) < int(n_qb/2):
                            if (n_qb % int(i * j) ) != 0:
                                elements_class_new.append(int(i * j) )
                                elements_class_new.append(n_qb - int(i * j) )
                                num_elements_class_new = num_elements_class_new + 2
                        else:
                            break
                    elements_class.append(elements_class_new)
                    num_elements_class.append(num_elements_class_new)
            elements_class.append([int(n_qb/2)] )   # elements of C2_i' are omitted
            if n_qb > 2:
                num_elements_class.append(1 + int(n_qb/2) )
                elements_class.append([] )          # elements of C2_i are omitted
                num_elements_class.append(int(n_qb/2) )
            else:
                num_elements_class.append(1)
        else:                   #if n_qb is odd
            # The classes are ordered to be e, Cn, ..., C2_i
            # record the index i of Cn^i
            elements_class.append([0] )
            num_elements_class.append(1)
            for i in range(1, int((n_qb+1)/2) ):
                if (n_qb % i) == 0:
                    elements_class_new = [i, int(n_qb - i)]
                    num_elements_class_new = 2
                    for j in range(2, n_qb):
                        if int(i * j) < int((n_qb+1)/2):
                            if (n_qb % int(i * j) ) != 0:
                                elements_class_new.append(int(i * j) )
                                elements_class_new.append(n_qb - int(i * j) )
                                num_elements_class_new = num_elements_class_new + 2
                        else:
                            break
                    elements_class.append(elements_class_new)
                    num_elements_class.append(num_elements_class_new)
            elements_class.append([] )          # elements of C2_i are omitted
            num_elements_class.append(n_qb)
        
        # Delete duplicate elements:
        if (n_qb % 2) == 0:     #if n_qb is even
            for i in range(0, len(num_elements_class)-2 ):
                for j in range(0, len(num_elements_class)-3-i ):
                    for k in (elements_class[int(-3-i)] ):
                        if (k in elements_class[j]):
                            elements_class[j].remove(k)
                            num_elements_class[j] = num_elements_class[j] - 1
        else:                   #if n_qb is odd
            for i in range(0, len(num_elements_class)-1 ):
                for j in range(0, len(num_elements_class)-2-i ):
                    for k in (elements_class[int(-2-i)] ):
                        if (k in elements_class[j]):
                            elements_class[j].remove(k)
                            num_elements_class[j] = num_elements_class[j] - 1
        
        # Generate the character table of S_n group, for elements in D_n only, n>=2:
        if n_qb == 2:
            character_table_Sn = np.array([ [1, 1], [1, -1] ], dtype=np.int64)
        else:
            character_table_Sn = np.zeros((len(young_diagram_irrep), len(num_elements_class) ), dtype=np.int64)
            if (n_qb % 2) == 0:     #if n_qb is even
                for i in range(0, len(num_elements_class) ):    # the one-dimensional irrep
                    character_table_Sn[0][i] = 1
                for i in range(0, len(young_diagram_irrep) ):   # the identity element
                    character_table_Sn[i][0] = int( (factorial(n_qb) * int(young_diagram_irrep[i][0] - young_diagram_irrep[i][1] + 1) ) /
                                                (factorial(int(young_diagram_irrep[i][0] + 1) ) * factorial(young_diagram_irrep[i][1] ) ) )
                for i in range(0, len(young_diagram_irrep) ):   # the C2_i class
                    if (young_diagram_irrep[i][1] % 2) == 0:
                        character_table_Sn[i][-1] = self.Catalan_dual(int((young_diagram_irrep[i][0]-2) / 2 ), int(young_diagram_irrep[i][1]/2), tilde=False)
                    else:
                        character_table_Sn[i][-1] = self.Catalan_dual(int((young_diagram_irrep[i][0]-1) / 2 ), int((young_diagram_irrep[i][1]-1) / 2), tilde=False)
                for i in range(1, len(num_elements_class)-2 ):    # the other classes except (C2_i' and Cn^(n/2) )
                    for j in range(1, len(young_diagram_irrep) ):
                        if int(young_diagram_irrep[j][1] % young_diagram_class[i][0] ) == 0:
                            character_table_Sn_j_i = self.Catalan_dual(int(young_diagram_irrep[j][0] / young_diagram_class[i][0] ), int(young_diagram_irrep[j][1] / young_diagram_class[i][0] ), tilde=True)
                            for k in range(1, int(young_diagram_irrep[j][1] / young_diagram_class[i][0] )+1 ):
                                character_table_Sn_j_i = character_table_Sn_j_i + self.modified_Catalan(k) * self.Catalan_dual(int((young_diagram_irrep[j][0] / young_diagram_class[i][0]) - k ), int((young_diagram_irrep[j][1] / young_diagram_class[i][0]) - k ), tilde=True)
                            character_table_Sn[j][i] = int(character_table_Sn_j_i)
                        elif int(young_diagram_irrep[j][1] % young_diagram_class[i][0] ) == 1:
                            character_table_Sn_j_i = self.Catalan_dual(int((young_diagram_irrep[j][0] + 1)/young_diagram_class[i][0] ), int((young_diagram_irrep[j][1]-1) / young_diagram_class[i][0] ), tilde=True)
                            for k in range(1, int((young_diagram_irrep[j][1]-1) / young_diagram_class[i][0] )+1 ):
                                character_table_Sn_j_i = character_table_Sn_j_i + self.modified_Catalan(k) * self.Catalan_dual(int(((young_diagram_irrep[j][0] + 1)/young_diagram_class[i][0]) - k ), int(((young_diagram_irrep[j][1]-1) / young_diagram_class[i][0]) - k ), tilde=True)
                            character_table_Sn_j_i = - character_table_Sn_j_i
                            character_table_Sn[j][i] = int(character_table_Sn_j_i)
                        else:
                            character_table_Sn[j][i] = 0
                for i in range(0, len(young_diagram_irrep) ):   # the (C2_i' and Cn^(n/2) ) class
                    if (young_diagram_irrep[i][1] % 2) == 0:
                        character_table_Sn_i_negative2 = self.Catalan_dual(int(young_diagram_irrep[i][0] / 2), int(young_diagram_irrep[i][1] / 2), tilde=True)
                        for k in range(1, int(young_diagram_irrep[i][1] / 2)+1 ):
                            character_table_Sn_i_negative2 = character_table_Sn_i_negative2 + self.modified_Catalan(k) * self.Catalan_dual(int((young_diagram_irrep[i][0] / 2) - k ), int((young_diagram_irrep[i][1] / 2) - k ), tilde=True)
                        character_table_Sn[i][-2] = int(character_table_Sn_i_negative2)
                    else:
                        character_table_Sn_i_negative2 = self.Catalan_dual(int((young_diagram_irrep[i][0] + 1) / 2), int((young_diagram_irrep[i][1]-1) / 2 ), tilde=True)
                        for k in range(1, int((young_diagram_irrep[i][1]-1) / 2)+1 ):
                            character_table_Sn_i_negative2 = character_table_Sn_i_negative2 + self.modified_Catalan(k) * self.Catalan_dual(int(((young_diagram_irrep[i][0] + 1) / 2) - k ), int(((young_diagram_irrep[i][1]-1) / 2) - k ), tilde=True)
                        character_table_Sn_i_negative2 = - character_table_Sn_i_negative2
                        character_table_Sn[i][-2] = int(character_table_Sn_i_negative2)
            else:                   #if n_qb is odd
                for i in range(0, len(num_elements_class) ):    # the one-dimensional irrep
                    character_table_Sn[0][i] = 1
                for i in range(0, len(young_diagram_irrep) ):   # the identity element
                    character_table_Sn[i][0] = int( (factorial(n_qb) * int(young_diagram_irrep[i][0] - young_diagram_irrep[i][1] + 1) ) /
                                                (factorial(int(young_diagram_irrep[i][0] + 1) ) * factorial(young_diagram_irrep[i][1] ) ) )
                for i in range(0, len(young_diagram_irrep) ):   # the C2_i class
                    if (young_diagram_irrep[i][1] % 2) == 0:
                        character_table_Sn[i][-1] = self.Catalan_dual(int((young_diagram_irrep[i][0]-1) / 2 ), int(young_diagram_irrep[i][1]/2), tilde=False)
                    else:
                        character_table_Sn[i][-1] = 0
                for i in range(1, len(num_elements_class)-1 ):    # the other classes
                    for j in range(1, len(young_diagram_irrep) ):
                        if int(young_diagram_irrep[j][1] % young_diagram_class[i][0] ) == 0:
                            character_table_Sn_j_i = self.Catalan_dual(int(young_diagram_irrep[j][0] / young_diagram_class[i][0] ), int(young_diagram_irrep[j][1] / young_diagram_class[i][0] ), tilde=True)
                            for k in range(1, int(young_diagram_irrep[j][1] / young_diagram_class[i][0] )+1 ):
                                character_table_Sn_j_i = character_table_Sn_j_i + self.modified_Catalan(k) * self.Catalan_dual(int((young_diagram_irrep[j][0] / young_diagram_class[i][0]) - k ), int((young_diagram_irrep[j][1] / young_diagram_class[i][0]) - k ), tilde=True)
                            character_table_Sn[j][i] = int(character_table_Sn_j_i)
                        elif int(young_diagram_irrep[j][1] % young_diagram_class[i][0] ) == 1:
                            character_table_Sn_j_i = self.Catalan_dual(int((young_diagram_irrep[j][0] + 1)/young_diagram_class[i][0] ), int((young_diagram_irrep[j][1]-1) / young_diagram_class[i][0] ), tilde=True)
                            for k in range(1, int((young_diagram_irrep[j][1]-1) / young_diagram_class[i][0] )+1 ):
                                character_table_Sn_j_i = character_table_Sn_j_i + self.modified_Catalan(k) * self.Catalan_dual(int(((young_diagram_irrep[j][0] + 1)/young_diagram_class[i][0]) - k ), int(((young_diagram_irrep[j][1]-1) / young_diagram_class[i][0]) - k ), tilde=True)
                            character_table_Sn_j_i = - character_table_Sn_j_i
                            character_table_Sn[j][i] = int(character_table_Sn_j_i)
                        else:
                            character_table_Sn[j][i] = 0
        
        return young_diagram_irrep, young_diagram_class, elements_class, num_elements_class, character_table_Sn
    
    def character_table_Dn(self, n_qb):
    # Output the character table of D_n.
        # Denote the irreps:
        if n_qb == 2:
            idx_irrep = [ [1], [1] ]
        else:
            if (n_qb % 2) == 0:     #if n_qb is even
                idx_irrep = [ [1], [1], [1], [1] ]    # the Id, C, D_(n/2)_(1), D_(n/2)_(2) irreps
                for i in range(1, int(n_qb/2) ):
                    idx_irrep.append([i, int(n_qb-i)])
            else:                   #if n_qb is odd
                idx_irrep = [ [1], [1] ]    # the Id, C irreps
                for i in range(1, int((n_qb+1)/2 ) ):
                    idx_irrep.append([i, int(n_qb-i)]) 
        
        # Generate the elements and number of elements in each class:
        # n_qb must >= 2
        elements_class = []
        num_elements_class = []
        if (n_qb % 2) == 0:     #if n_qb is even
            elements_class.append([0] )
            num_elements_class.append(1)
            for i in range(1, int(n_qb/2) ):
                elements_class.append([i, int(n_qb-i)])
                num_elements_class.append(2)
            elements_class.append([int(n_qb/2)] )
            num_elements_class.append(1)
            if n_qb > 2:
                elements_class.append([] )   # elements of C2_i' are omitted
                num_elements_class.append(int(n_qb/2) )
                elements_class.append([] )          # elements of C2_i are omitted
                num_elements_class.append(int(n_qb/2) )
        else:                   #if n_qb is odd
            elements_class.append([0] )
            num_elements_class.append(1)
            for i in range(1, int((n_qb+1)/2 ) ):
                elements_class.append([i, int(n_qb-i)])
                num_elements_class.append(2)
            elements_class.append([] )          # elements of C2_i are omitted
            num_elements_class.append(n_qb)
        
        # Generate the character table of D_n group, n>=2:
        if n_qb == 2:
            character_table_Dn = np.array([ [1.0, 1.0], [1.0, -1.0] ], dtype=np.float64)
        else:
            character_table_Dn = np.zeros((len(idx_irrep), len(num_elements_class) ), dtype=np.float64)
            if (n_qb % 2) == 0:     #if n_qb is even
                for i in range(0, len(num_elements_class) ):    # the Id irrep
                    character_table_Dn[0][i] = 1.0
                for i in range(0, len(num_elements_class)-2 ):    # the C irrep
                    character_table_Dn[1][i] = 1.0
                character_table_Dn[1][-1] = - 1.0
                character_table_Dn[1][-2] = - 1.0
                for i in range(0, len(num_elements_class)-2 ):    # the D_(n/2)_(1) irrep
                    if (i % 2) == 0:
                        character_table_Dn[2][i] = 1.0
                    else:
                        character_table_Dn[2][i] = - 1.0
                character_table_Dn[2][-1] = - 1.0
                character_table_Dn[2][-2] = 1.0
                for i in range(0, len(num_elements_class)-2 ):    # the D_(n/2)_(2) irrep
                    if (i % 2) == 0:
                        character_table_Dn[3][i] = 1.0
                    else:
                        character_table_Dn[3][i] = - 1.0
                character_table_Dn[3][-1] = 1.0
                character_table_Dn[3][-2] = - 1.0
                for i in range(4, len(idx_irrep) ):               # the two-dimensional irreps
                    for j in range(0, len(num_elements_class)-2 ):
                        if (int(n_qb % 4) == 0) and (int(idx_irrep[i][0] * elements_class[j][0]) % int(n_qb / 2) == int(n_qb / 4) ):
                            character_table_Dn[i][j] = 0.0
                        else:
                            character_table_Dn[i][j] = 2.0 * np.cos((2.0 * pi / float(n_qb) ) * float(idx_irrep[i][0] * elements_class[j][0] ) )
                    #character_table_Dn[i][-1] = 0.0
                    #character_table_Dn[i][-2] = 0.0
            else:                   #if n_qb is odd
                for i in range(0, len(num_elements_class) ):    # the Id irrep
                    character_table_Dn[0][i] = 1.0
                for i in range(0, len(num_elements_class)-1 ):    # the C irrep
                    character_table_Dn[1][i] = 1.0
                character_table_Dn[1][-1] = -1.0
                for i in range(2, len(idx_irrep) ):               # the two-dimensional irreps
                    for j in range(0, len(num_elements_class)-1 ):
                        character_table_Dn[i][j] = 2.0 * np.cos((2.0 * pi / float(n_qb) ) * float(idx_irrep[i][0] * elements_class[j][0] ) )
                    #character_table_Dn[i][-1] = 0.0
        
        return idx_irrep, elements_class, num_elements_class, character_table_Dn
    
    def Catalan_single(self, a, tilde):
    # returns the single-element Catalan number, a is positive integer
    # The way to travel from the bottom-left point to the top-right point
    # with reaching but not crossing the diagonal (tilde = False)
    # or, without reaching the diagonal (tilde = True)
    # in one a * a lattice
        if tilde == False:
        # catalan_single(a, tilde = False) = catalan_dual(a, a, tilde = False)
            catalan_single = int(comb(int(a*2), a, exact=True) / (a+1) )
        else:
        # catalan_single(a, tilde = True) = catalan_single(a-1, tilde = False)
            if a == 1:
                catalan_single = 1
            else:
                catalan_single = int(comb(int((a-1)*2), (a-1), exact=True) / a)
        
        return catalan_single
    
    def Catalan_dual(self, a, b, tilde):
    # returns the dual-element Catalan number, it must be satisfied that a > b, a and b are positive integers
    # The way to travel from the bottom-left point to the top-right point
    # with reaching but not crossing the diagonal (tilde = False)
    # or, without reaching the diagonal (tilde = True)
    # in one a * b lattice
    # Catalan_dual(a, 0) = 1
    # Catalan_dual(a, 1) = a
        if b == 0:
            catalan_dual = 1
        elif b == 1:
            if tilde == False:
                catalan_dual = a
            else:
                catalan_dual = a - 1
        else:
            if tilde == False:
                catalan_dual = int(comb(int(a+b), b, exact=True) - comb(int(a+b), int(b-1), exact=True) )
            else:
            # catalan_dual(a, b, tilde = True) = catalan_dual(a-1, b, tilde = False)
                catalan_dual = int(comb(int(a+b-1), b, exact=True) - comb(int(a+b-1), int(b-1), exact=True) )
        
        return catalan_dual
    
    def modified_Catalan(self, a):
    # Returns the value of following function:
    # Sum_over_i (n_i)! / (PI_over_j (n_i_j)! ) * (Pi_over_l Catalan~(e_i_l) ) * 2^(n_i)
    # a is positive integer
    # Decompose a into addition of positive integers. List all possible decompositions, each denoted as i
    # e.g., a = 4, a = 4 = 3 + 1 = 2 + 2 = 2 + 1 + 1 = 1 + 1 + 1 + 1
    # The array will be [ [4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1] ]
    # n_i is the length (number of integers) of the i'th array [1, 2, 2, 3, 4]
    # n_i_j is the number of times that each distinctive integer appears [ [1], [1, 1], [2], [1, 2], [4] ]
    # e_i_l is the l'th element of the i'th array
    # the term 2^(n_i) counts for the number of all regular and modified filling
        partition = self.integer_partition(a)
        integer_list = []
        integer_count_list = []
        for i in range(0, len(partition) ):
            integer_list_i = []
            integer_count_list_i = []
            for j in range(1, a+1):
                if (j in partition[i] ):
                    integer_list_i.append(j)
                    integer_count_list_i.append(partition[i].count(j) )
            integer_list.append(integer_list_i)
            integer_count_list.append(integer_count_list_i)
        
        modified_catalan = 0
        for i in range(0, len(partition) ):
            length_i = len(partition[i] )
            modified_catalan_i = factorial(length_i)
            for j in range(0, len(integer_count_list[i]) ):
                modified_catalan_i = modified_catalan_i / factorial(integer_count_list[i][j] )
            for j in range(0, length_i):
                modified_catalan_i = modified_catalan_i * self.Catalan_single(partition[i][j], tilde=True)
            modified_catalan_i = modified_catalan_i * (2**length_i)
            modified_catalan = int(modified_catalan + modified_catalan_i)
        
        return modified_catalan
    
    def integer_partition(self, n):
    # Output all possible partitions of positive integer n:
    # code from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
        partition_chain = []
        partition_chain.append([ () ] )
        partition_chain.append([ (1, ) ] )   # (1, ) is a tuple, not integer
        for i in range(2, n+1):
            partition_i = set()
            for j in range(0, i):
                for partition in partition_chain[j]:
                    partition_i.add(tuple(sorted((i - j, ) + partition) ) )
            partition_chain.append(list(partition_i))
        
        partition_n = partition_chain[n]
        partition_n_list = []
        for i in partition_n:
            partition_n_list.append(list(i) )
        
        return partition_n_list
    
    def Young_method_Sn(self, n_qb):
    # Output Q_full, Q_dagger_full, J_subspace, J_eigenstate, M_eigenstate with Young method when there is no coupling term
        young_tableau, young_diagram, dim_irrep, young_tableau_chain, young_diagram_chain, dim_irrep_chain = self.Young_tableau(n_qb)
        weyl_tableau, weyl_tableau_num, basis_for_diag_operator = self.Weyl_tableau(n_qb, young_tableau, young_diagram, dim_irrep)
        diag_unitary_irrep_operator, flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays, diag_unitary_irrep_operator_coefficient_frac, diag_unitary_irrep_operator_sign = self.Young_operator(n_qb, young_tableau_chain, young_diagram_chain, dim_irrep_chain)
        Q_full, Q_dagger_full, J_subspace, J_eigenstate, M_eigenstate = self.solve_for_Adjoint_Q_Young_method(n_qb, 
             young_diagram, dim_irrep, weyl_tableau_num, basis_for_diag_operator, diag_unitary_irrep_operator, flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays, diag_unitary_irrep_operator_coefficient_frac, diag_unitary_irrep_operator_sign)
        
        return Q_full, Q_dagger_full, J_subspace, J_eigenstate, M_eigenstate
        
    def Young_tableau(self, n_qb):
    # output all one-row and two-row standard Young tableaux for S_n group, n = n_qb, n_qb >= 2
    # as well as the shape of Young diagram and the dimension of irreps
    # as well as the chain of them for i = 2, 3, 4, ..., n_qb
        young_tableau = [ [[1, 2]], [[1], [2]] ]
        young_tableau_chain = [ [ [[1, 2]], [[1], [2]] ] ]
        young_diagram = [ [2], [1, 1] ]
        young_diagram_chain = [ [ [2], [1, 1] ] ]
        dim_irrep = [1, 1]    # dimension of irrep equals to the number of standard Young tableaux
        dim_irrep_chain = [ [1, 1] ]
        
        if n_qb == 2:
            return young_tableau, young_diagram, dim_irrep, young_tableau_chain, young_diagram_chain, dim_irrep_chain
        else:
            for i in range(3, n_qb+1):
                if (i % 2) == 0:     #if i is even
                    # The first two Young tableaux:
                    young_tableau_new = []
                    young_tableau_one_row = young_tableau[0][0] + [i]   # [1, 2, ..., i-1, i]
                    young_tableau_new.append([young_tableau_one_row])
                    young_tableau_new.append([young_tableau[0][0], [i]])   # [ [1, 2, ..., i-1], [i] ]
                    # Other Young tableaux:
                    idx = 1
                    for j in range(1, len(dim_irrep) ):
                        # add [i] to  the end of the first row:
                        for k in range(idx, int(idx + dim_irrep[j]) ):
                            young_tableau_two_row = young_tableau[k]
                            young_tableau_two_row_0 = young_tableau_two_row[0] + [i]
                            young_tableau_two_row_new = [young_tableau_two_row_0, young_tableau_two_row[1] ]
                            young_tableau_new.append(young_tableau_two_row_new)
                        # add [i] to  the end of the second row:
                        for k in range(idx, int(idx + dim_irrep[j]) ):
                            young_tableau_two_row = young_tableau[k]
                            young_tableau_two_row_1 = young_tableau_two_row[1] + [i]
                            young_tableau_two_row_new = [young_tableau_two_row[0], young_tableau_two_row_1 ]
                            young_tableau_new.append(young_tableau_two_row_new)
                        idx = idx + dim_irrep[j]
                    
                    # The Young diagrams:
                    young_diagram_new = [[i]]
                    for j in range(1, int(i/2)+1 ):
                        young_diagram_new.append([i-j, j])
                    
                    # The dimension of irreps:
                    dim_irrep_new = [1]
                    for j in range(1, len(dim_irrep) ):
                        dim_irrep_new.append(int(dim_irrep[j-1] + dim_irrep[j]) )
                    dim_irrep_new.append(dim_irrep[-1])
                                   
                else:                   #if i is odd
                    young_tableau_new = []
                    young_tableau_one_row = young_tableau[0][0] + [i]
                    young_tableau_new.append([young_tableau_one_row])
                    young_tableau_new.append([young_tableau[0][0], [i]])
                    
                    idx = 1
                    for j in range(1, len(dim_irrep)-1 ):
                        for k in range(idx, int(idx + dim_irrep[j]) ):
                            young_tableau_two_row = young_tableau[k]
                            young_tableau_two_row_0 = young_tableau_two_row[0] + [i]
                            young_tableau_two_row_new = [young_tableau_two_row_0, young_tableau_two_row[1] ]
                            young_tableau_new.append(young_tableau_two_row_new)
                        for k in range(idx, int(idx + dim_irrep[j]) ):
                            young_tableau_two_row = young_tableau[k]
                            young_tableau_two_row_1 = young_tableau_two_row[1] + [i]
                            young_tableau_two_row_new = [young_tableau_two_row[0], young_tableau_two_row_1 ]
                            young_tableau_new.append(young_tableau_two_row_new)
                        idx = idx + dim_irrep[j]
                    # When i is odd, [i] is added to the end of the first row only in the last update to Young tableaux
                    for k in range(idx, int(idx + dim_irrep[-1]) ):
                        young_tableau_two_row = young_tableau[k]
                        young_tableau_two_row_0 = young_tableau_two_row[0] + [i]
                        young_tableau_two_row_new = [young_tableau_two_row_0, young_tableau_two_row[1] ]
                        young_tableau_new.append(young_tableau_two_row_new)
                    
                    young_diagram_new = [[i]]
                    for j in range(1, int((i-1)/2)+1 ):
                        young_diagram_new.append([i-j, j])
                    
                    dim_irrep_new = [1]
                    for j in range(1, len(dim_irrep) ):
                        dim_irrep_new.append(int(dim_irrep[j-1] + dim_irrep[j]) )
                
                young_tableau = young_tableau_new
                young_diagram = young_diagram_new
                dim_irrep = dim_irrep_new
                young_tableau_chain.append(young_tableau)
                young_diagram_chain.append(young_diagram)
                dim_irrep_chain.append(dim_irrep)
            # young_tableau is not in dictionary order here, but the Young tableaux of each Young diagram are put together    
            return young_tableau, young_diagram, dim_irrep, young_tableau_chain, young_diagram_chain, dim_irrep_chain
    
    def Weyl_tableau(self, n_qb, young_tableau, young_diagram, dim_irrep):
    # Output all Weyl tableaux of each Young diagram, and the basis of each Young operator
        weyl_tableau = []
        weyl_tableau_num = []
        weyl_tableau_num.append(int(n_qb+1) )
        
        # weyl_tableau: 0 for spin-up, 1 for spin-down
        weyl_tableau_one_row = []   # all Weyl tableaux for one-row diag operator
        for i in range(0, int(n_qb+1) ):
            weyl_tableau_one_row_i = [ [0] * int(n_qb-i) + [1] * (n_qb - int(n_qb-i) ) ]
            weyl_tableau_one_row.append(weyl_tableau_one_row_i)
        weyl_tableau.append(weyl_tableau_one_row)
        
        # weyl_tableau_num is the dimension of Weyl tableaux
        # Calculate for the Weyl tableaux for two-row diag operators
        for i in range(1, len(dim_irrep) ):
            weyl_tableau_num.append(int(young_diagram[i][0] - young_diagram[i][1] + 1) )
            
            weyl_tableau_two_row_i = []
            for j in range(0, int(young_diagram[i][0] - young_diagram[i][1] + 1) ):
                weyl_tableau_two_row_i_j = []                
                weyl_tableau_two_row_i_j_0 = [0] * int(young_diagram[i][0] - j ) + [1] * (young_diagram[i][0] - int(young_diagram[i][0] - j ) )
                weyl_tableau_two_row_i_j.append(weyl_tableau_two_row_i_j_0)                
                weyl_tableau_two_row_i_j_1 = [1] * young_diagram[i][1]
                weyl_tableau_two_row_i_j.append(weyl_tableau_two_row_i_j_1)              
                weyl_tableau_two_row_i.append(weyl_tableau_two_row_i_j)
                
            weyl_tableau.append(weyl_tableau_two_row_i)
        
        # The diag operators will operate on basis_for_young_operator:
        basis_for_young_operator = []
        # The following part can be omitted because the basis for the first operator are exactly the same with the weyl_tableau[0]
        basis_for_young_operator_0 = []
        young_tableau_0 = young_tableau[0]
        weyl_tableau_0 = weyl_tableau[0]
        for i in range(0, weyl_tableau_num[0]):     # rearrange the order of Weyl tableau in that of the Young tableau
            young_tableau_0_one_dim = young_tableau_0[0]
            weyl_tableau_0_one_dim = weyl_tableau_0[i][0]
            basis_for_young_operator_0_i = [0] * n_qb
            for j in range(0, n_qb):
                basis_for_young_operator_0_i[ young_tableau_0_one_dim[j]-1 ] = weyl_tableau_0_one_dim[j]
            '''
            basis_for_young_operator_0_i = []
            for j in range(1, int(n_qb+1) ):
                for k in range(0, n_qb):
                    if young_tableau_0_one_dim[k] == j:
                        basis_for_young_operator_0_i.append(weyl_tableau_0_one_dim[k])
                        break
            '''
            basis_for_young_operator_0.append(basis_for_young_operator_0_i)
        basis_for_young_operator.append(basis_for_young_operator_0)
        
        # basis for two=-row diag operators:
        idx = 1
        for i in range(1, len(dim_irrep) ):
            weyl_tableau_i = weyl_tableau[i]
            for j in range(0, dim_irrep[i]):
                young_tableau_idx = young_tableau[idx]
                basis_for_young_operator_idx = []
                young_tableau_idx_one_dim = young_tableau_idx[0] + young_tableau_idx[1]
                for k in range(0, weyl_tableau_num[i]):
                    weyl_tableau_i_k = weyl_tableau_i[k]
                    weyl_tableau_i_k_one_dim = weyl_tableau_i_k[0] + weyl_tableau_i_k[1]
                    basis_for_young_operator_idx_k = [0] * n_qb
                    for l in range(0, n_qb):
                        basis_for_young_operator_idx_k[ young_tableau_idx_one_dim[l]-1 ] = weyl_tableau_i_k_one_dim[l]
                    '''
                    basis_for_young_operator_idx_k = []
                    for l in range(1, int(n_qb+1) ):
                        for m in range(0, n_qb):
                            if young_tableau_idx_one_dim[m] == l:
                                basis_for_young_operator_idx_k.append(weyl_tableau_i_k_one_dim[m])
                                break
                    '''
                    basis_for_young_operator_idx.append(basis_for_young_operator_idx_k)
                basis_for_young_operator.append(basis_for_young_operator_idx)
                idx = idx + 1
                           
        return weyl_tableau, weyl_tableau_num, basis_for_young_operator
    
    def Young_operator(self, n_qb, young_tableau_chain, young_diagram_chain, dim_irrep_chain):
    # Output the Young operator and the operator of Sum(diagonal_of_unitary_irrep * group_element)
        first_row_operator = [ [[1, 2], [2, 1]], [[1]] ]
        # List all arrays of the first row after each permutation
        second_row_operator = [ [], [[2]] ]
        # List all arrays of the second row after each permutation
        col_operator = [ [], [[], [1, 2]] ]
        # List all transpositions among two rows, but keep in mind that a permutation can be made up of multiple transpositions
        comb_col = [ [], [[], [1]] ]
        comb_col_sign = [ [1], [1, 0] ]
        # all possible combination of col_operator and the parity
        # 1 for plus, 0 for minus
        young_operator = [ [[1, 2], [2, 1]], [[1, 2], [2, 1]] ]
        # List all elements of the Young operator
        # Each element is denoted as [1, 2, ..., n] [n1, n2, ..., n_n], the first array is omitted
        young_operator_sign = [ [1, 1], [1, 0] ]
        # 1 for plus, 0 for minus
        diag_unitary_irrep_operator = [ [[1, 2], [2, 1]], [] ]
        flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays = [1, 1]
        diag_unitary_irrep_operator_coefficient_frac = [ [[1, 2], [1, 2]], [[1, 2], [1, 2]] ]
        # [numerator, denominator]
        diag_unitary_irrep_operator_sign = [ [1, 1], [1, 0] ]
        # 1 for plus, 0 for minus
        
        # The coefficients are recorded in fractional format for precision.
        # They are converted to decimals only in the last step of solving for Q.
        
        # Generate young_operator and diag_unitary_irrep_operator in iterative way:
        if n_qb == 2:
            return diag_unitary_irrep_operator, flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays, diag_unitary_irrep_operator_coefficient_frac, diag_unitary_irrep_operator_sign
        else:
            for i in range(3, int(n_qb+1) ):
                dim_irrep_i_minus_1 = copy.deepcopy(dim_irrep_chain[i-3] )
                dim_irrep_i = copy.deepcopy(dim_irrep_chain[i-2] )
                young_diagram = copy.deepcopy(young_diagram_chain[i-3] )
                young_tableau = copy.deepcopy(young_tableau_chain[i-3] )
                if (i % 2) == 0:        #if i is even
                    first_row_operator_new = []
                    second_row_operator_new = [ [], [[i]] ]
                    col_operator_new = [ [], [[], [1, i]] ]
                    young_operator_new = []
                    young_operator_sign_new = []
                    diag_unitary_irrep_operator_new = []
                    flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new = [1]
                    diag_unitary_irrep_operator_coefficient_frac_new = []
                    diag_unitary_irrep_operator_sign_new = []
                    
                    # Generate the Young operators for the one-row and the first two-row Young tableaux:
                    first_row_operator_one_row = []
                    for j in range(0, factorial(i-1) ):
                        first_row_operator_one_row.append(first_row_operator[0][j] + [i] )
                    young_operator_two_row_0 = copy.deepcopy(first_row_operator_one_row)
                    for j in range(0, factorial(i-1) ):
                        young_operator_two_row_0.append([i] + first_row_operator[0][j][1:] + [first_row_operator[0][j][0] ] )
                    young_operator_sign_one_row = [1] * factorial(i)
                    young_operator_sign_two_row_0 = young_operator_sign[0] + [0] * factorial(i-1)
                    for j in range(i-2, -1, -1):
                        for k in range(0, factorial(i-1) ):
                            first_row_operator_0_k = first_row_operator[0][k]
                            first_row_operator_one_row.append(first_row_operator_0_k[:j] + [i] + first_row_operator_0_k[j:] )
                    first_row_operator_new.append(first_row_operator_one_row)
                    first_row_operator_new.append(first_row_operator[0])
                    young_operator_new.append(first_row_operator_one_row)
                    young_operator_sign_new.append(young_operator_sign_one_row)
                    young_operator_new.append(young_operator_two_row_0)
                    young_operator_sign_new.append(young_operator_sign_two_row_0)
                    
                    # generate the diag operator with Young operator and the last diag operator in the group chain:
                    diag_unitary_irrep_operator_coefficient_frac_one_row = []
                    for j in range(0, factorial(i) ):
                        diag_unitary_irrep_operator_coefficient_frac_one_row.append([1, factorial(i)])
                    diag_unitary_irrep_operator_new.append(first_row_operator_one_row)
                    diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_one_row)
                    diag_unitary_irrep_operator_sign_new.append(young_operator_sign_one_row)
                    diag_unitary_irrep_operator_two_row_0, diag_unitary_irrep_operator_coefficient_frac_two_row_0, diag_unitary_irrep_operator_sign_two_row_0, flag_diag_operator_two_row_0_have_all_possible_permutation_arrays = self.young_2_diag_op(i, dim_irrep_i[1], 
                        young_operator_two_row_0, young_operator_sign_two_row_0, diag_unitary_irrep_operator[0], diag_unitary_irrep_operator_coefficient_frac[0], diag_unitary_irrep_operator_sign[0], 
                        first_row_operator_one_row, evalDiagOpTwoRow0 = True)
                    diag_unitary_irrep_operator_new.append(diag_unitary_irrep_operator_two_row_0)
                    flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new.append(flag_diag_operator_two_row_0_have_all_possible_permutation_arrays)
                    diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_two_row_0)
                    diag_unitary_irrep_operator_sign_new.append(diag_unitary_irrep_operator_sign_two_row_0)
                    
                    # comb_col and comb_col_sign are renewed only when i is even
                    comb_col_new = copy.deepcopy(comb_col[int(i/2)-1] )
                    comb_col_sign_new = copy.deepcopy(comb_col_sign[int(i/2)-1] )
                    for j in range(0, int(2**(int(i/2)-1) ) ):
                        comb_col_new.append(comb_col[int(i/2)-1][j] + [int(i/2)])
                        if comb_col_sign[int(i/2)-1][j] == 1:
                            comb_col_sign_new.append(0)
                        else:
                            comb_col_sign_new.append(1)
                    comb_col.append(comb_col_new)
                    comb_col_sign.append(comb_col_sign_new)
                    
                    # Generate the Young operators and the diag operators for other two-row Young tableaux:
                    # The structure of for loops here are very similar to that in Young_tableau.
                    idx = 1
                    for j in range(1, len(dim_irrep_i_minus_1) ):
                        young_diagram_j = copy.deepcopy(young_diagram[j] )
                        for k in range(idx, int(idx + dim_irrep_i_minus_1[j]) ):
                            young_tableau_k = copy.deepcopy(young_tableau[k] )
                            first_row_operator_k = copy.deepcopy(first_row_operator[k] )
                            first_row_operator_two_row = []
                            for l in range(young_diagram_j[0], -1, -1):
                                for m in range(0, factorial(young_diagram_j[0]) ):
                                    first_row_operator_k_m = copy.deepcopy(first_row_operator_k[m] )
                                    first_row_operator_two_row.append(first_row_operator_k_m[:l] + [i] + first_row_operator_k_m[l:] )
                            first_row_operator_new.append(first_row_operator_two_row)
                            second_row_operator_new.append(second_row_operator[k])
                            col_operator_new.append(col_operator[k])
                            young_diagram_two_row = [young_diagram_j[0]+1, young_diagram_j[1] ]
                            young_tableau_two_row = [(young_tableau_k[0] + [i]), young_tableau_k[1] ]
                            young_operator_two_row, young_operator_sign_two_row = self.row_col_2_young_op(i, 
                                first_row_operator_two_row, second_row_operator[k], col_operator[k], young_diagram_two_row, young_tableau_two_row, comb_col, comb_col_sign)
                            young_operator_new.append(young_operator_two_row)
                            young_operator_sign_new.append(young_operator_sign_two_row)
                            if flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays[k] == 1:
                                diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[0] )
                            else:
                                diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[k] )
                            diag_unitary_irrep_operator_two_row, diag_unitary_irrep_operator_coefficient_frac_two_row, diag_unitary_irrep_operator_sign_two_row, flag_diag_operator_two_row_have_all_possible_permutation_arrays = self.young_2_diag_op(i, dim_irrep_i[j], 
                                young_operator_two_row, young_operator_sign_two_row, diag_unitary_irrep_operator_k, diag_unitary_irrep_operator_coefficient_frac[k], diag_unitary_irrep_operator_sign[k], 
                                first_row_operator_one_row, evalDiagOpTwoRow0 = False)
                            diag_unitary_irrep_operator_new.append(diag_unitary_irrep_operator_two_row)
                            flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new.append(flag_diag_operator_two_row_have_all_possible_permutation_arrays)
                            diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_two_row)
                            diag_unitary_irrep_operator_sign_new.append(diag_unitary_irrep_operator_sign_two_row)
                            
                        for k in range(idx, int(idx + dim_irrep_i_minus_1[j]) ):
                            young_tableau_k = copy.deepcopy(young_tableau[k] )
                            first_row_operator_new.append(first_row_operator[k])
                            second_row_operator_k = copy.deepcopy(second_row_operator[k] )
                            second_row_operator_two_row = []
                            for l in range(young_diagram_j[1], -1, -1):
                                for m in range(0, factorial(young_diagram_j[1]) ):
                                    second_row_operator_k_m = copy.deepcopy(second_row_operator_k[m] )
                                    second_row_operator_two_row.append(second_row_operator_k_m[:l] + [i] + second_row_operator_k_m[l:] )
                            second_row_operator_new.append(second_row_operator_two_row)
                            col_operator_k = copy.deepcopy(col_operator[k])
                            col_operator_k.append([young_tableau_k[0][young_diagram_j[1] ] ,i])
                            col_operator_new.append(col_operator_k)
                            young_diagram_two_row = [young_diagram_j[0], young_diagram_j[1]+1 ]
                            young_tableau_two_row = [young_tableau_k[0], (young_tableau_k[1] + [i]) ]
                            young_operator_two_row, young_operator_sign_two_row = self.row_col_2_young_op(i, 
                                first_row_operator[k], second_row_operator_two_row, col_operator_k, young_diagram_two_row, young_tableau_two_row, comb_col, comb_col_sign)
                            young_operator_new.append(young_operator_two_row)
                            young_operator_sign_new.append(young_operator_sign_two_row)
                            if flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays[k] == 1:
                                diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[0] )
                            else:
                                diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[k] )
                            diag_unitary_irrep_operator_two_row, diag_unitary_irrep_operator_coefficient_frac_two_row, diag_unitary_irrep_operator_sign_two_row, flag_diag_operator_two_row_have_all_possible_permutation_arrays = self.young_2_diag_op(i, dim_irrep_i[j+1], 
                                young_operator_two_row, young_operator_sign_two_row, diag_unitary_irrep_operator_k, diag_unitary_irrep_operator_coefficient_frac[k], diag_unitary_irrep_operator_sign[k], 
                                first_row_operator_one_row, evalDiagOpTwoRow0 = False)
                            diag_unitary_irrep_operator_new.append(diag_unitary_irrep_operator_two_row)
                            flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new.append(flag_diag_operator_two_row_have_all_possible_permutation_arrays)
                            diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_two_row)
                            diag_unitary_irrep_operator_sign_new.append(diag_unitary_irrep_operator_sign_two_row)
                        idx = idx + dim_irrep_i_minus_1[j]
                            
                else:                   #if i is odd
                    first_row_operator_new = []
                    second_row_operator_new = [ [], [[i]] ]
                    col_operator_new = [ [], [[], [1, i]] ]
                    young_operator_new = []
                    young_operator_sign_new = []
                    diag_unitary_irrep_operator_new = []
                    flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new = [1]
                    diag_unitary_irrep_operator_coefficient_frac_new = []
                    diag_unitary_irrep_operator_sign_new = []
                    
                    first_row_operator_one_row = []
                    for j in range(0, factorial(i-1) ):
                        first_row_operator_one_row.append(first_row_operator[0][j] + [i] )
                    young_operator_two_row_0 = copy.deepcopy(first_row_operator_one_row)
                    for j in range(0, factorial(i-1) ):
                        young_operator_two_row_0.append([i] + first_row_operator[0][j][1:] + [first_row_operator[0][j][0] ] )
                    young_operator_sign_one_row = [1] * factorial(i)
                    young_operator_sign_two_row_0 = young_operator_sign[0] + [0] * factorial(i-1)
                    for j in range(i-2, -1, -1):
                        for k in range(0, factorial(i-1) ):
                            first_row_operator_0_k = first_row_operator[0][k]
                            first_row_operator_one_row.append(first_row_operator_0_k[:j] + [i] + first_row_operator_0_k[j:] )
                    first_row_operator_new.append(first_row_operator_one_row)
                    first_row_operator_new.append(first_row_operator[0])
                    young_operator_new.append(first_row_operator_one_row)
                    young_operator_sign_new.append(young_operator_sign_one_row)
                    young_operator_new.append(young_operator_two_row_0)
                    young_operator_sign_new.append(young_operator_sign_two_row_0)
                    diag_unitary_irrep_operator_coefficient_frac_one_row = []
                    for j in range(0, factorial(i) ):
                        diag_unitary_irrep_operator_coefficient_frac_one_row.append([1, factorial(i)])
                    diag_unitary_irrep_operator_new.append(first_row_operator_one_row)
                    diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_one_row)
                    diag_unitary_irrep_operator_sign_new.append(young_operator_sign_one_row)
                    diag_unitary_irrep_operator_two_row_0, diag_unitary_irrep_operator_coefficient_frac_two_row_0, diag_unitary_irrep_operator_sign_two_row_0, flag_diag_operator_two_row_0_have_all_possible_permutation_arrays = self.young_2_diag_op(i, dim_irrep_i[1], 
                        young_operator_two_row_0, young_operator_sign_two_row_0, diag_unitary_irrep_operator[0], diag_unitary_irrep_operator_coefficient_frac[0], diag_unitary_irrep_operator_sign[0], 
                        first_row_operator_one_row, evalDiagOpTwoRow0 = True)
                    diag_unitary_irrep_operator_new.append(diag_unitary_irrep_operator_two_row_0)
                    flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new.append(flag_diag_operator_two_row_0_have_all_possible_permutation_arrays)
                    diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_two_row_0)
                    diag_unitary_irrep_operator_sign_new.append(diag_unitary_irrep_operator_sign_two_row_0)
                    
                    idx = 1
                    for j in range(1, len(dim_irrep_i_minus_1)-1 ):
                        young_diagram_j = copy.deepcopy(young_diagram[j] )
                        for k in range(idx, int(idx + dim_irrep_i_minus_1[j]) ):
                            young_tableau_k = copy.deepcopy(young_tableau[k] )
                            first_row_operator_k = copy.deepcopy(first_row_operator[k] )
                            first_row_operator_two_row = []
                            for l in range(young_diagram_j[0], -1, -1):
                                for m in range(0, factorial(young_diagram_j[0]) ):
                                    first_row_operator_k_m = copy.deepcopy(first_row_operator_k[m] )
                                    first_row_operator_two_row.append(first_row_operator_k_m[:l] + [i] + first_row_operator_k_m[l:] )
                            first_row_operator_new.append(first_row_operator_two_row)
                            second_row_operator_new.append(second_row_operator[k])
                            col_operator_new.append(col_operator[k])
                            young_diagram_two_row = [young_diagram_j[0]+1, young_diagram_j[1] ]
                            young_tableau_two_row = [(young_tableau_k[0] + [i]), young_tableau_k[1] ]
                            young_operator_two_row, young_operator_sign_two_row = self.row_col_2_young_op(i, 
                                first_row_operator_two_row, second_row_operator[k], col_operator[k], young_diagram_two_row, young_tableau_two_row, comb_col, comb_col_sign)
                            young_operator_new.append(young_operator_two_row)
                            young_operator_sign_new.append(young_operator_sign_two_row)
                            if flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays[k] == 1:
                                diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[0] )
                            else:
                                diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[k] )
                            diag_unitary_irrep_operator_two_row, diag_unitary_irrep_operator_coefficient_frac_two_row, diag_unitary_irrep_operator_sign_two_row, flag_diag_operator_two_row_have_all_possible_permutation_arrays = self.young_2_diag_op(i, dim_irrep_i[j], 
                                young_operator_two_row, young_operator_sign_two_row, diag_unitary_irrep_operator_k, diag_unitary_irrep_operator_coefficient_frac[k], diag_unitary_irrep_operator_sign[k], 
                                first_row_operator_one_row, evalDiagOpTwoRow0 = False)
                            diag_unitary_irrep_operator_new.append(diag_unitary_irrep_operator_two_row)
                            flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new.append(flag_diag_operator_two_row_have_all_possible_permutation_arrays)
                            diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_two_row)
                            diag_unitary_irrep_operator_sign_new.append(diag_unitary_irrep_operator_sign_two_row)
                            
                        for k in range(idx, int(idx + dim_irrep_i_minus_1[j]) ):
                            young_tableau_k = copy.deepcopy(young_tableau[k] )
                            first_row_operator_new.append(first_row_operator[k])
                            second_row_operator_k = copy.deepcopy(second_row_operator[k] )
                            second_row_operator_two_row = []
                            for l in range(young_diagram_j[1], -1, -1):
                                for m in range(0, factorial(young_diagram_j[1]) ):
                                    second_row_operator_k_m = copy.deepcopy(second_row_operator_k[m] )
                                    second_row_operator_two_row.append(second_row_operator_k_m[:l] + [i] + second_row_operator_k_m[l:] )
                            second_row_operator_new.append(second_row_operator_two_row)
                            col_operator_k = copy.deepcopy(col_operator[k])
                            col_operator_k.append([young_tableau_k[0][young_diagram_j[1] ] ,i])
                            col_operator_new.append(col_operator_k)
                            young_diagram_two_row = [young_diagram_j[0], young_diagram_j[1]+1 ]
                            young_tableau_two_row = [young_tableau_k[0], (young_tableau_k[1] + [i]) ]
                            young_operator_two_row, young_operator_sign_two_row = self.row_col_2_young_op(i, 
                                first_row_operator[k], second_row_operator_two_row, col_operator_k, young_diagram_two_row, young_tableau_two_row, comb_col, comb_col_sign)
                            young_operator_new.append(young_operator_two_row)
                            young_operator_sign_new.append(young_operator_sign_two_row)
                            if flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays[k] == 1:
                                diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[0] )
                            else:
                                diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[k] )
                            diag_unitary_irrep_operator_two_row, diag_unitary_irrep_operator_coefficient_frac_two_row, diag_unitary_irrep_operator_sign_two_row, flag_diag_operator_two_row_have_all_possible_permutation_arrays = self.young_2_diag_op(i, dim_irrep_i[j+1], 
                                young_operator_two_row, young_operator_sign_two_row, diag_unitary_irrep_operator_k, diag_unitary_irrep_operator_coefficient_frac[k], diag_unitary_irrep_operator_sign[k], 
                                first_row_operator_one_row, evalDiagOpTwoRow0 = False)
                            diag_unitary_irrep_operator_new.append(diag_unitary_irrep_operator_two_row)
                            flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new.append(flag_diag_operator_two_row_have_all_possible_permutation_arrays)
                            diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_two_row)
                            diag_unitary_irrep_operator_sign_new.append(diag_unitary_irrep_operator_sign_two_row)
                        idx = idx + dim_irrep_i_minus_1[j]
                        
                    for k in range(idx, int(idx + dim_irrep_i_minus_1[-1]) ):
                        young_tableau_k = copy.deepcopy(young_tableau[k] )
                        first_row_operator_k = copy.deepcopy(first_row_operator[k] )
                        first_row_operator_two_row = []
                        for l in range(young_diagram[-1][0], -1, -1):
                            for m in range(0, factorial(young_diagram[-1][0]) ):
                                first_row_operator_k_m = copy.deepcopy(first_row_operator_k[m] )
                                first_row_operator_two_row.append(first_row_operator_k_m[:l] + [i] + first_row_operator_k_m[l:] )
                        first_row_operator_new.append(first_row_operator_two_row)
                        second_row_operator_new.append(second_row_operator[k])
                        col_operator_new.append(col_operator[k])
                        young_diagram_two_row = [young_diagram[-1][0]+1, young_diagram[-1][1] ]
                        young_tableau_two_row = [(young_tableau_k[0] + [i]), young_tableau_k[1] ]
                        young_operator_two_row, young_operator_sign_two_row = self.row_col_2_young_op(i, 
                            first_row_operator_two_row, second_row_operator[k], col_operator[k], young_diagram_two_row, young_tableau_two_row, comb_col, comb_col_sign)
                        young_operator_new.append(young_operator_two_row)
                        young_operator_sign_new.append(young_operator_sign_two_row)
                        if flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays[k] == 1:
                            diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[0] )
                        else:
                            diag_unitary_irrep_operator_k = copy.deepcopy(diag_unitary_irrep_operator[k] )
                        diag_unitary_irrep_operator_two_row, diag_unitary_irrep_operator_coefficient_frac_two_row, diag_unitary_irrep_operator_sign_two_row, flag_diag_operator_two_row_have_all_possible_permutation_arrays = self.young_2_diag_op(i, dim_irrep_i[-1], 
                            young_operator_two_row, young_operator_sign_two_row, diag_unitary_irrep_operator_k, diag_unitary_irrep_operator_coefficient_frac[k], diag_unitary_irrep_operator_sign[k], 
                            first_row_operator_one_row, evalDiagOpTwoRow0 = False)
                        diag_unitary_irrep_operator_new.append(diag_unitary_irrep_operator_two_row)
                        flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new.append(flag_diag_operator_two_row_have_all_possible_permutation_arrays)
                        diag_unitary_irrep_operator_coefficient_frac_new.append(diag_unitary_irrep_operator_coefficient_frac_two_row)
                        diag_unitary_irrep_operator_sign_new.append(diag_unitary_irrep_operator_sign_two_row)
                        
                first_row_operator = copy.deepcopy(first_row_operator_new)
                second_row_operator = copy.deepcopy(second_row_operator_new)
                col_operator = copy.deepcopy(col_operator_new)
                young_operator = copy.deepcopy(young_operator_new)
                young_operator_sign = copy.deepcopy(young_operator_sign_new)
                diag_unitary_irrep_operator = copy.deepcopy(diag_unitary_irrep_operator_new)
                flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays = copy.deepcopy(flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays_new)
                diag_unitary_irrep_operator_coefficient_frac = copy.deepcopy(diag_unitary_irrep_operator_coefficient_frac_new)
                diag_unitary_irrep_operator_sign = copy.deepcopy(diag_unitary_irrep_operator_sign_new)
            
            return diag_unitary_irrep_operator, flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays, diag_unitary_irrep_operator_coefficient_frac, diag_unitary_irrep_operator_sign
    
    def row_col_2_young_op(self, max_int, first_row_operator, second_row_operator, col_operator, young_diagram_two_row, young_tableau_two_row, comb_col, comb_col_sign):
    # Convert the row and column operators to Young operator for young_diagram[1] >= 1
        young_operator_two_row = []
        young_operator_sign_two_row = []
        comb_col_two_row = copy.deepcopy(comb_col[young_diagram_two_row[1] ] )
        comb_col_sign_two_row = copy.deepcopy(comb_col_sign[young_diagram_two_row[1] ] )
        
        # Output Young operator elements when column operator is identity (one-row Young tableaux):
        young_tableau_one_dim = young_tableau_two_row[0] + young_tableau_two_row[1]
        for j in range(0, factorial(young_diagram_two_row[0]) ):
            for k in range(0, factorial(young_diagram_two_row[1]) ):
                permutation_element_one_dim = first_row_operator[j] + second_row_operator[k]
                young_operator_element_new = [0] * max_int
                for l in range(0, max_int):
                    young_operator_element_new[ young_tableau_one_dim[l]-1 ] = permutation_element_one_dim[l]
                '''
                young_operator_element_new = []
                for l in range(1, int(max_int+1) ):
                    for m in range(0, max_int):
                        if young_tableau_one_dim[m] == l:
                            young_operator_element_new.append(permutation_element_one_dim[m])
                            break
                '''
                young_operator_two_row.append(young_operator_element_new)
                young_operator_sign_two_row.append(1)
        
        # Output Young operator elements when Young tableaux are two-row:
        for i in range(1, len(comb_col_two_row) ):
            col_operator_indicator = copy.deepcopy(comb_col_two_row[i] )
            col_operator_sign = copy.deepcopy(comb_col_sign_two_row[i] )
            for j in range(0, factorial(young_diagram_two_row[0]) ):
                for k in range(0, factorial(young_diagram_two_row[1]) ):
                    young_tableau_two_row_0 = copy.deepcopy(young_tableau_two_row[0] )
                    young_tableau_two_row_1 = copy.deepcopy(young_tableau_two_row[1] )
                    permutation_element_one_dim = first_row_operator[j] + second_row_operator[k]
                    for l in range(0, len(col_operator_indicator) ):
                        col_operator_l = col_operator[col_operator_indicator[l] ]
                        young_tableau_two_row_0[col_operator_indicator[l] - 1 ] = col_operator_l[1]
                        young_tableau_two_row_1[col_operator_indicator[l] - 1 ] = col_operator_l[0]
                    young_tableau_one_dim = young_tableau_two_row_0 + young_tableau_two_row_1
                    young_operator_element_new = [0] * max_int
                    for l in range(0, max_int):
                        young_operator_element_new[ young_tableau_one_dim[l]-1 ] = permutation_element_one_dim[l]
                    '''
                    young_operator_element_new = []
                    for l in range(1, int(max_int+1) ):
                        for m in range(0, max_int):
                            if young_tableau_one_dim[m] == l:
                                young_operator_element_new.append(permutation_element_one_dim[m])
                                break
                    '''
                    young_operator_two_row.append(young_operator_element_new)
                    young_operator_sign_two_row.append(col_operator_sign)
            
        return young_operator_two_row, young_operator_sign_two_row
    
    def young_2_diag_op(self, max_int, dim_irrep, young_operator, young_operator_sign, diag_operator, diag_operator_coefficient_frac, diag_operator_sign, all_possible_permutation_arrays, evalDiagOpTwoRow0):
    # Convert the Young operator to the operator of Sum(diagonal_of_unitary_irrep * group_element)
        gcd = np.gcd(dim_irrep, factorial(max_int) )
        numerator = int(dim_irrep / gcd)
        denominator = int(factorial(max_int) / gcd)
        
        diag_operator_two_row = copy.deepcopy(all_possible_permutation_arrays )
        diag_operator_two_row_max_possible_length = factorial(max_int)
        diag_operator_coefficient_frac_two_row = []
        for i in range(0, diag_operator_two_row_max_possible_length):
            diag_operator_coefficient_frac_two_row.append([0, 1])
        diag_operator_sign_two_row = [1] * diag_operator_two_row_max_possible_length
        
        # Calculate for diag_operator_two_row = Young * diag
        for i in range(0, len(diag_operator_sign) ):
            diag_operator_i = diag_operator[i] + [max_int]
            for j in range(0, len(young_operator_sign) ):
                young_operator_j = copy.deepcopy(young_operator[j] )
                diag_operator_two_row_element = []
                for k in diag_operator_i:
                    diag_operator_two_row_element.append(young_operator_j[k-1])
                '''
                for k in range(0, max_int):
                    for l in range(1, int(max_int+1) ):
                        if diag_operator_i[k] == l:
                            diag_operator_two_row_element.append(young_operator_j[l-1])
                            break
                '''
                diag_operator_coefficient_frac_two_row_element = [ diag_operator_coefficient_frac[i][0] * numerator,
                                                                  diag_operator_coefficient_frac[i][1] * denominator ]
                if (diag_operator_sign[i] == 1) and (young_operator_sign[j] == 1):
                    diag_operator_sign_two_row_element = 1
                elif (diag_operator_sign[i] == 0) and (young_operator_sign[j] == 0):
                    diag_operator_sign_two_row_element = 1
                else:
                    diag_operator_sign_two_row_element = 0
                for k in range(0, diag_operator_two_row_max_possible_length):
                    if diag_operator_two_row[k] == diag_operator_two_row_element:
                        diag_operator_coefficient_frac_two_row_k = copy.deepcopy(diag_operator_coefficient_frac_two_row[k] )
                        diag_operator_sign_two_row_k = copy.deepcopy(diag_operator_sign_two_row[k] )
                        # Add diag_operator_coefficient_frac_two_row_element to diag_operator_coefficient_frac_two_row_k
                        denominator_new = int(diag_operator_coefficient_frac_two_row_k[1] * diag_operator_coefficient_frac_two_row_element[1] )
                        if (diag_operator_sign_two_row_k == 1) and (diag_operator_sign_two_row_element == 1):
                            numerator_new = int( diag_operator_coefficient_frac_two_row_k[0] * diag_operator_coefficient_frac_two_row_element[1] + 
                                diag_operator_coefficient_frac_two_row_k[1] * diag_operator_coefficient_frac_two_row_element[0] )
                            diag_operator_sign_two_row[k] = 1
                        elif (diag_operator_sign_two_row_k == 0) and (diag_operator_sign_two_row_element == 0):
                            numerator_new = int( diag_operator_coefficient_frac_two_row_k[0] * diag_operator_coefficient_frac_two_row_element[1] + 
                                diag_operator_coefficient_frac_two_row_k[1] * diag_operator_coefficient_frac_two_row_element[0] )
                            diag_operator_sign_two_row[k] = 0
                        elif (diag_operator_sign_two_row_k == 1) and (diag_operator_sign_two_row_element == 0):
                            numerator_new = int( diag_operator_coefficient_frac_two_row_k[0] * diag_operator_coefficient_frac_two_row_element[1] - 
                                diag_operator_coefficient_frac_two_row_k[1] * diag_operator_coefficient_frac_two_row_element[0] )
                            if numerator_new >= 0:
                                diag_operator_sign_two_row[k] = 1
                            else:
                                numerator_new = - numerator_new
                                diag_operator_sign_two_row[k] = 0
                        else:
                            numerator_new = int( - diag_operator_coefficient_frac_two_row_k[0] * diag_operator_coefficient_frac_two_row_element[1] + 
                                diag_operator_coefficient_frac_two_row_k[1] * diag_operator_coefficient_frac_two_row_element[0] )
                            if numerator_new >= 0:
                                diag_operator_sign_two_row[k] = 1
                            else:
                                numerator_new = - numerator_new
                                diag_operator_sign_two_row[k] = 0
                        if numerator_new == 0:
                            diag_operator_coefficient_frac_two_row[k][0] = 0
                            diag_operator_coefficient_frac_two_row[k][1] = 1
                        else:
                            gcd = np.gcd(numerator_new, denominator_new)
                            diag_operator_coefficient_frac_two_row[k][0] = int(numerator_new / gcd)
                            diag_operator_coefficient_frac_two_row[k][1] = int(denominator_new / gcd)
                        break
        
        flag_diag_operator_two_row_0_have_all_possible_permutation_arrays = 1
        for i in range(int(diag_operator_two_row_max_possible_length-1), -1, -1):
            if diag_operator_coefficient_frac_two_row[i][0] == 0:
                flag_diag_operator_two_row_0_have_all_possible_permutation_arrays = 0
                del diag_operator_two_row[i]
                del diag_operator_coefficient_frac_two_row[i]
                del diag_operator_sign_two_row[i]
        
        # No need to calculate for diag * Young when evalDiagOpTwoRow0 == True
        if evalDiagOpTwoRow0 == True:
            # No need to record diag_operator_two_row if it have all possible permutation arrays:
            if flag_diag_operator_two_row_0_have_all_possible_permutation_arrays == 1:
                diag_operator_two_row = []
            return diag_operator_two_row, diag_operator_coefficient_frac_two_row, diag_operator_sign_two_row, flag_diag_operator_two_row_0_have_all_possible_permutation_arrays
        else:
            diag_operator_two_row_new = copy.deepcopy(all_possible_permutation_arrays )
            diag_operator_coefficient_frac_two_row_new = []
            for i in range(0, diag_operator_two_row_max_possible_length):
                diag_operator_coefficient_frac_two_row_new.append([0, 1])
            diag_operator_sign_two_row_new = [1] * diag_operator_two_row_max_possible_length            
            
            # Calculate for diag_operator_two_row_new = diag * diag_operator_two_row:
            for i in range(0, len(diag_operator_sign) ):
                diag_operator_i = diag_operator[i] + [max_int]
                for j in range(0, len(diag_operator_sign_two_row) ):
                    diag_operator_two_row_j = copy.deepcopy(diag_operator_two_row[j] )
                    diag_operator_two_row_new_element = []
                    for k in diag_operator_two_row_j:
                        diag_operator_two_row_new_element.append(diag_operator_i[k-1])
                    '''
                    for k in range(0, max_int):
                        for l in range(1, int(max_int+1) ):
                            if diag_operator_two_row_j[k] == l:
                                diag_operator_two_row_new_element.append(diag_operator_i[l-1])
                                break
                    '''
                    diag_operator_coefficient_frac_two_row_new_element = [ diag_operator_coefficient_frac[i][0] * diag_operator_coefficient_frac_two_row[j][0], 
                                                                          diag_operator_coefficient_frac[i][1] * diag_operator_coefficient_frac_two_row[j][1] ]
                    if (diag_operator_sign[i] == 1) and (diag_operator_sign_two_row[j] == 1):
                        diag_operator_sign_two_row_new_element = 1
                    elif (diag_operator_sign[i] == 0) and (diag_operator_sign_two_row[j] == 0):
                        diag_operator_sign_two_row_new_element = 1
                    else:
                        diag_operator_sign_two_row_new_element = 0
                    for k in range(0, diag_operator_two_row_max_possible_length):
                        if diag_operator_two_row_new[k] == diag_operator_two_row_new_element:
                            diag_operator_coefficient_frac_two_row_new_k = copy.deepcopy(diag_operator_coefficient_frac_two_row_new[k] )
                            diag_operator_sign_two_row_new_k = copy.deepcopy(diag_operator_sign_two_row_new[k] )
                            denominator_new = int(diag_operator_coefficient_frac_two_row_new_k[1] * diag_operator_coefficient_frac_two_row_new_element[1] )
                            if (diag_operator_sign_two_row_new_k == 1) and (diag_operator_sign_two_row_new_element == 1):
                                numerator_new = int( diag_operator_coefficient_frac_two_row_new_k[0] * diag_operator_coefficient_frac_two_row_new_element[1] + 
                                    diag_operator_coefficient_frac_two_row_new_k[1] * diag_operator_coefficient_frac_two_row_new_element[0] )
                                diag_operator_sign_two_row_new[k] = 1
                            elif (diag_operator_sign_two_row_new_k == 0) and (diag_operator_sign_two_row_new_element == 0):
                                numerator_new = int( diag_operator_coefficient_frac_two_row_new_k[0] * diag_operator_coefficient_frac_two_row_new_element[1] + 
                                    diag_operator_coefficient_frac_two_row_new_k[1] * diag_operator_coefficient_frac_two_row_new_element[0] )
                                diag_operator_sign_two_row_new[k] = 0
                            elif (diag_operator_sign_two_row_new_k == 1) and (diag_operator_sign_two_row_new_element == 0):
                                numerator_new = int( diag_operator_coefficient_frac_two_row_new_k[0] * diag_operator_coefficient_frac_two_row_new_element[1] - 
                                    diag_operator_coefficient_frac_two_row_new_k[1] * diag_operator_coefficient_frac_two_row_new_element[0] )
                                if numerator_new >= 0:
                                    diag_operator_sign_two_row_new[k] = 1
                                else:
                                    numerator_new = - numerator_new
                                    diag_operator_sign_two_row_new[k] = 0
                            else:
                                numerator_new = int( - diag_operator_coefficient_frac_two_row_new_k[0] * diag_operator_coefficient_frac_two_row_new_element[1] + 
                                    diag_operator_coefficient_frac_two_row_new_k[1] * diag_operator_coefficient_frac_two_row_new_element[0] )
                                if numerator_new >= 0:
                                    diag_operator_sign_two_row_new[k] = 1
                                else:
                                    numerator_new = - numerator_new
                                    diag_operator_sign_two_row_new[k] = 0
                            
                            if numerator_new == 0:
                                diag_operator_coefficient_frac_two_row_new[k][0] = 0
                                diag_operator_coefficient_frac_two_row_new[k][1] = 1
                            else:
                                gcd = np.gcd(numerator_new, denominator_new)
                                diag_operator_coefficient_frac_two_row_new[k][0] = int(numerator_new / gcd)
                                diag_operator_coefficient_frac_two_row_new[k][1] = int(denominator_new / gcd)
                            break
            
            flag_diag_operator_two_row_new_have_all_possible_permutation_arrays = 1
            for i in range(int(diag_operator_two_row_max_possible_length-1), -1, -1):
                if diag_operator_coefficient_frac_two_row_new[i][0] == 0:
                    flag_diag_operator_two_row_new_have_all_possible_permutation_arrays = 0
                    del diag_operator_two_row_new[i]
                    del diag_operator_coefficient_frac_two_row_new[i]
                    del diag_operator_sign_two_row_new[i]
            
            if flag_diag_operator_two_row_new_have_all_possible_permutation_arrays == 1:
                diag_operator_two_row_new = []
            return diag_operator_two_row_new, diag_operator_coefficient_frac_two_row_new, diag_operator_sign_two_row_new, flag_diag_operator_two_row_new_have_all_possible_permutation_arrays
    
    def solve_for_Adjoint_Q_Young_method(self, n_qb, young_diagram, dim_irrep, weyl_tableau_num, basis_for_diag_operator, diag_unitary_irrep_operator, flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays, diag_unitary_irrep_operator_coefficient_frac, diag_unitary_irrep_operator_sign):
    # Output the adjoint matrix Q, Q_dagger, J and M with Young method
        # generate the one-hot basis, 0 for spin-up, 1 for spin-down:   
        all_possible_basis = [ [[0, 0]], [[0, 1], [1, 0]], [[1, 1]] ]
        if n_qb > 2:
            for i in range(3, int(n_qb+1) ):
                all_possible_basis_new = []
                all_possible_basis_new_0 = [ [0] * i ]
                all_possible_basis_new.append(all_possible_basis_new_0)
                for j in range(1, i):
                    all_possible_basis_new_j = []
                    for k in range(0, comb(i-1, j-1, exact=True) ):
                        all_possible_basis_new_j.append(all_possible_basis[j-1][k] + [1] )
                    for k in range(0, comb(i-1, j, exact=True) ):
                        all_possible_basis_new_j.append(all_possible_basis[j][k] + [0] )
                    all_possible_basis_new.append(all_possible_basis_new_j)
                all_possible_basis_new_n_qb = [ [1] * i ]
                all_possible_basis_new.append(all_possible_basis_new_n_qb)
                all_possible_basis = copy.deepcopy(all_possible_basis_new)
        #print(all_possible_basis)
        
        # Each base will be a one-hot array, idx_all_possible_basis list the index of 1 for each array:
        idx_all_possible_basis = []
        idx_all_possible_basis_0 = [0]
        idx_all_possible_basis.append(idx_all_possible_basis_0)
        for i in range(1, n_qb):
            idx_all_possible_basis_i = []
            all_possible_basis_i = copy.deepcopy(all_possible_basis[i] )
            for j in range(0, comb(n_qb, i, exact=True) ):
                all_possible_basis_i_j = copy.deepcopy(all_possible_basis_i[j] )
                idx = 0
                for k in range(0, n_qb):
                    if all_possible_basis_i_j[k] == 1:
                        idx = idx + int(2**int(n_qb - 1 - k) )
                idx_all_possible_basis_i.append(idx)
            idx_all_possible_basis.append(idx_all_possible_basis_i)
        idx_all_possible_basis_n_qb = [ int(2**n_qb - 1) ]
        idx_all_possible_basis.append(idx_all_possible_basis_n_qb)
        #print(idx_all_possible_basis)
        
        base_coefficient_frac = []
        base_sign = []
        # Generate the principle subspace (J = n_qb/2) :
        base_coefficient_frac_0 = []
        base_sign_0 = []
        base_coefficient_frac_0_0 = [[1, 1]]   # operator 0 on base 0
        base_sign_0_0 = [1]   # 1 for plus, 0 for minus
        base_coefficient_frac_0.append(base_coefficient_frac_0_0)
        base_sign_0.append(base_sign_0_0)
        for i in range(1, n_qb):
            base_coefficient_frac_0_i = [[1, comb(n_qb, i, exact=True) ]] * comb(n_qb, i, exact=True)
            base_sign_0_i = [1] * comb(n_qb, i, exact=True)
            base_coefficient_frac_0.append(base_coefficient_frac_0_i)
            base_sign_0.append(base_sign_0_i)
        base_coefficient_frac_0_n_qb = [[1, 1]]
        base_sign_0_n_qb = [1]
        base_coefficient_frac_0.append(base_coefficient_frac_0_n_qb)
        base_sign_0.append(base_sign_0_n_qb)
        base_coefficient_frac.append(base_coefficient_frac_0)
        base_sign.append(base_sign_0)
        #print(base_coefficient_frac_0)
        #print(base_sign_0)
        
        # Generate other subspaces:
        idx = 1
        for i in range(1, len(dim_irrep) ):
            for j in range(idx, int(idx + dim_irrep[i]) ):
                if flag_diag_unitary_irrep_operator_have_all_possible_permutation_arrays[j] == 1:
                    diag_unitary_irrep_operator_j = copy.deepcopy(diag_unitary_irrep_operator[0] )
                else:
                    diag_unitary_irrep_operator_j = copy.deepcopy(diag_unitary_irrep_operator[j] )
                diag_unitary_irrep_operator_coefficient_frac_j = copy.deepcopy(diag_unitary_irrep_operator_coefficient_frac[j] )
                diag_unitary_irrep_operator_sign_j = copy.deepcopy(diag_unitary_irrep_operator_sign[j] )
                basis_for_diag_operator_j = copy.deepcopy(basis_for_diag_operator[j] )
                base_coefficient_frac_j = []
                base_sign_j = []
                for k in range(0, weyl_tableau_num[i] ):
                    basis_for_diag_operator_j_k = copy.deepcopy(basis_for_diag_operator_j[k] )
                    all_possible_basis_j_k = copy.deepcopy(all_possible_basis[int(young_diagram[i][1] + k)] )
                    base_coefficient_frac_j_k = []
                    for l in range(0, comb(n_qb, int(young_diagram[i][1] + k), exact=True) ):
                        base_coefficient_frac_j_k.append([0, 1])
                    base_sign_j_k = [1] * comb(n_qb, int(young_diagram[i][1] + k), exact=True)
                    for l in range(0, len(diag_unitary_irrep_operator_sign_j) ):
                        diag_unitary_irrep_operator_j_l = copy.deepcopy(diag_unitary_irrep_operator_j[l] )
                        diag_unitary_irrep_operator_coefficient_frac_j_l = copy.deepcopy(diag_unitary_irrep_operator_coefficient_frac_j[l] )
                        diag_unitary_irrep_operator_sign_j_l = copy.deepcopy(diag_unitary_irrep_operator_sign_j[l] )
                        # operate the diag operator element on the Weyl base:
                        basis_for_diag_operator_new = [0] * n_qb
                        for m in range(0, n_qb):
                            basis_for_diag_operator_new[ diag_unitary_irrep_operator_j_l[m]-1 ] = basis_for_diag_operator_j_k[m]
                        '''
                        basis_for_diag_operator_new = []
                        for m in range(1, int(n_qb+1) ):
                            for n in range(0, n_qb):
                                if diag_unitary_irrep_operator_j_l[n] == m:
                                    basis_for_diag_operator_new.append(basis_for_diag_operator_j_k[n])
                                    break
                        '''
                        # Add diag_unitary_irrep_operator_coefficient_frac_j_l to base_coefficient_frac_j_k :
                        for m in range(0, comb(n_qb, int(young_diagram[i][1] + k), exact=True) ):
                            if all_possible_basis_j_k[m] == basis_for_diag_operator_new:
                                denominator_new = int(diag_unitary_irrep_operator_coefficient_frac_j_l[1] * base_coefficient_frac_j_k[m][1] )
                                if (base_sign_j_k[m] == 1) and (diag_unitary_irrep_operator_sign_j_l == 1):
                                    numerator_new = int( base_coefficient_frac_j_k[m][0] * diag_unitary_irrep_operator_coefficient_frac_j_l[1] + 
                                                        base_coefficient_frac_j_k[m][1] * diag_unitary_irrep_operator_coefficient_frac_j_l[0] )
                                    base_sign_j_k[m] == 1
                                elif (base_sign_j_k[m] == 0) and (diag_unitary_irrep_operator_sign_j_l == 0):
                                    numerator_new = int( base_coefficient_frac_j_k[m][0] * diag_unitary_irrep_operator_coefficient_frac_j_l[1] + 
                                                        base_coefficient_frac_j_k[m][1] * diag_unitary_irrep_operator_coefficient_frac_j_l[0] )
                                    base_sign_j_k[m] = 0
                                elif (base_sign_j_k[m] == 1) and (diag_unitary_irrep_operator_sign_j_l == 0):
                                    numerator_new = int( base_coefficient_frac_j_k[m][0] * diag_unitary_irrep_operator_coefficient_frac_j_l[1] - 
                                                        base_coefficient_frac_j_k[m][1] * diag_unitary_irrep_operator_coefficient_frac_j_l[0] )
                                    if numerator_new >= 0:
                                        base_sign_j_k[m] = 1
                                    else:
                                        numerator_new = - numerator_new
                                        base_sign_j_k[m] = 0
                                else:
                                    numerator_new = int( - base_coefficient_frac_j_k[m][0] * diag_unitary_irrep_operator_coefficient_frac_j_l[1] + 
                                                        base_coefficient_frac_j_k[m][1] * diag_unitary_irrep_operator_coefficient_frac_j_l[0] )
                                    if numerator_new >= 0:
                                        base_sign_j_k[m] = 1
                                    else:
                                        numerator_new = - numerator_new
                                        base_sign_j_k[m] = 0
                                if numerator_new == 0:
                                    base_coefficient_frac_j_k[m][0] = 0
                                    base_coefficient_frac_j_k[m][1] = 1
                                else:
                                    gcd = np.gcd(numerator_new, denominator_new)
                                    base_coefficient_frac_j_k[m][0] = int(numerator_new / gcd)
                                    base_coefficient_frac_j_k[m][1] = int(denominator_new / gcd)
                                break
                    base_coefficient_frac_j.append(base_coefficient_frac_j_k)
                    base_sign_j.append(base_sign_j_k)
                base_coefficient_frac.append(base_coefficient_frac_j)
                base_sign.append(base_sign_j)
            idx = int(idx + dim_irrep[i])
        #print(base_coefficient_frac)
        #print(base_sign)
        
        # Generate Q and Q_dagger by converting fractional numbers to decimal numbers:
        Q_dagger = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        idx = 0
        base_idx = 0
        for i in range(0, len(dim_irrep) ):
            for j in range(idx, int(idx + dim_irrep[i]) ):
                base_coefficient_frac_j = copy.deepcopy(base_coefficient_frac[j] )
                base_sign_j = copy.deepcopy(base_sign[j])
                for k in range(0, weyl_tableau_num[i] ):
                    base_coefficient_frac_j_k = copy.deepcopy(base_coefficient_frac_j[k] )
                    base_sign_j_k = copy.deepcopy(base_sign_j[k])
                    idx_all_possible_basis_j_k = copy.deepcopy(idx_all_possible_basis[int(i+k)] )
                    for l in range(0, comb(n_qb, int(i+k), exact=True) ):
                        if base_coefficient_frac_j_k[l][0] != 0:
                            if base_sign_j_k[l] == 1:
                                Q_dagger[base_idx][idx_all_possible_basis_j_k[l]] = float(base_coefficient_frac_j_k[l][0] ) / float(base_coefficient_frac_j_k[l][1] )
                            else:
                                Q_dagger[base_idx][idx_all_possible_basis_j_k[l]] = - float(base_coefficient_frac_j_k[l][0] ) / float(base_coefficient_frac_j_k[l][1] )
                    base_idx = base_idx + 1
            idx = int(idx + dim_irrep[i])
            
        for i in range(0, 2**n_qb):
            Q_dagger[i] = Q_dagger[i] / np.sqrt(np.sum(Q_dagger[i]**2) )
            
        Q = np.transpose(Q_dagger)
        #print(Q_dagger)
        #print(Q)
        
        J_subspace = []
        J = n_qb / 2.0
        for i in range(0, len(dim_irrep) ):
            for j in range(0, dim_irrep[i] ):
                J_subspace.append(J)
            J = J - 1.0
        
        J_eigenstate = []
        M_eigenstate = []
        J = n_qb / 2.0
        M = copy.deepcopy(J)
        for i in range(0, len(dim_irrep) ):
            for j in range(0, dim_irrep[i] ):
                M = copy.deepcopy(J)
                for k in range(0, weyl_tableau_num[i] ):
                    J_eigenstate.append(J)
                    M_eigenstate.append(M)
                    M = M - 1.0
            J = J - 1.0
            M = copy.deepcopy(J)
            
        J_subspace = np.array(J_subspace, dtype=np.float64)
        J_eigenstate = np.array(J_eigenstate, dtype=np.float64)
        M_eigenstate = np.array(M_eigenstate, dtype=np.float64)
        
        return Q, Q_dagger, J_subspace, J_eigenstate, M_eigenstate
    
    def solve_for_Adjoint_Q_CG_method(self, n_qb):
    # solve for adjoint matrix Q constructed with |J2, ..., J_i, J_n, M> basis with Clebsch-Gordan coefficients of SU(2)
        J = []             # quantum number J in the progress of adding spins
        M = []             # quantum number M in the progress of adding spins
        J_M = []           # quantum number [J, M]
        count_for_J_chain = []     # number of offsprings J evolve to 
        num = 0            # num evolves from 0 to number of qubits
        J_num = np.array([1/2], dtype=np.float64)    # all possible values of J at current num of qubits
        M_num = np.array([1/2, -1/2], dtype=np.float64)   # all possible values of M at current num of qubits
        J_M_num = np.array([[1/2, 1/2], [1/2, -1/2]], dtype=np.float64)    # all possible values of [J, M] at current num of qubits
        count_for_J_chain_num = np.array([2], dtype=np.int64)         # number of branches of J at current num of qubits
        J.append(J_num)
        M.append(M_num)
        J_M.append(J_M_num)
        count_for_J_chain.append(count_for_J_chain_num)     # number of offspring subspaces for each J
        for num in range(1, n_qb):
            J_num_new = []
            M_num_new = []
            J_M_num_new = []
            count_for_J_chain_num_new = []
            # calculate for all possible values of J when a new spin is added:
            for i in range( len(J_num) ):
                if J_num[i] == 0:
                    J_num_new.append(1/2)
                    count_for_J_chain_num_new.append(2)
                else:
                    J_num_new.append(J_num[i] + 1/2)
                    J_num_new.append(J_num[i] - 1/2)
                    count_for_J_chain_num_new.append(int(J_num[i] * 2 + 2) )
                    count_for_J_chain_num_new.append(int(J_num[i] * 2) )
            #calculate for all possible values of M when a new spin is added:
            for i in range( len(J_num_new) ):
                if J_num_new[i] == 0:
                    M_num_new.append(0)
                    J_M_num_new.append([0, 0])
                else:
                    M_num_new_append = J_num_new[i]
                    while not(M_num_new_append < -J_num_new[i]):
                        M_num_new.append(M_num_new_append)
                        J_M_num_new.append([J_num_new[i], M_num_new_append])
                        M_num_new_append = M_num_new_append - 1
            J.append(np.array(J_num_new) )
            M.append(np.array(M_num_new) )
            J_M.append(np.array(J_M_num_new) )
            count_for_J_chain.append(np.array(count_for_J_chain_num_new) )
            J_num = J_num_new
        
        for i in range(n_qb):
            count_for_J_chain[i] = count_for_J_chain[i] * (2**(n_qb - 1 - i) ) # number of offspring states for each J
        
        # Output analytical value of elements in Q_dagger:
        # Q_dagger_ifNonZero * Q_dagger_sign * sqrt(Q_dagger_frac[0] / Q_dagger_frac[1] )
        # Calculate for Q_dagger first because the |J, M> basis are in rows
        Q_dagger = np.zeros((2**n_qb, 2**n_qb), dtype=np.float64)
        Q_dagger_ifNonZero = np.ones((2**n_qb, 2**n_qb), dtype=np.int8)
        Q_dagger_sign = np.ones((2**n_qb, 2**n_qb), dtype=np.int8)
        Q_dagger_frac = np.ones((2**n_qb, 2**n_qb, 2), dtype=np.int64)
        # chain means the evolution record of ms, M, J, J_M
        ms_chain = np.zeros((2**n_qb, n_qb), dtype=np.float64)
        M_chain = np.zeros((2**n_qb, n_qb), dtype=np.float64)
        J_chain = np.zeros((2**n_qb, n_qb), dtype=np.float64)
        J_M_chain = np.zeros((2**n_qb, 2**n_qb, n_qb, 2), dtype=np.float64)
        
        # When calculating for Q_dagger, actually we are calculating for the probalility that the state evolves to some |J_i, M_i> from |J_i-1, M_i-1>
        # all arrangements of ms:
        for col in range(2**n_qb):
            col_4_compare = col         # use col because each column will be M
            for i in range(n_qb):
                if col_4_compare < 2**(n_qb - 1 - i):
                    ms_chain[col][i] = 1/2
                else:
                    ms_chain[col][i] = -1/2
                    col_4_compare = col_4_compare - 2**(n_qb - 1 - i)
        
        # M_chain is addition of ms at each number of qubits
        for col in range(2**n_qb):
            M_chain_col_i = 0.0
            for i in range(n_qb):
                M_chain_col_i = M_chain_col_i + ms_chain[col][i]
                M_chain[col][i] = M_chain_col_i
        
        # J_chain is the evolution progress of J till num of qubits for each eigenstate:
        for i in range(n_qb):
            row = 0
            count_num = 0
            while (row < (2**n_qb) ):
                for count in range(count_for_J_chain[n_qb - 1 - i][count_num]):   # J_chain is filled from last column to first column
                    J_chain[row][n_qb - 1 - i] = J[n_qb - 1 - i][count_num]
                    row = row + 1
                count_num = count_num + 1
        
        # pairing J and M:
        for row in range(2**n_qb):
            for col in range(2**n_qb):
                for i in range(n_qb):
                    J_M_chain[row][col][i] = [J_chain[row][i], M_chain[col][i] ]
        
        # correction for M in last J_M pair:
        for row in range(2**n_qb):
            for col in range(2**n_qb):
                J_M_chain[row][col][n_qb - 1][1] = M[-1][row]
        
        # element in Q_dagger is generalized Clebsch-Gordan coefficient <ms1, ms2, ..., ms_n | J2, J3, …, J_n, M>
        # this generalized Clebsch-Gordan coefficientis the product of series of Clebsch-Gordan coefficient
        # Calculate for analytical expression of Q_dagger:
        # Place (ms1, ms2, ..., ms_i, ..., ms_n) in rows, 2^n arrangements, m1 = m2 = ... = m_n = 1/2
        # Place (J1, M1; J2, M2; ...; J_i, M_i; ...; J_n, M_n) in columns, 2^n possible values
        # Calculate < J1, M1; J2, M2; ...; J_i, M_i; ...; J_n, M_n | ms1, ms2, ..., ms_i, ..., ms_n >, this is the value of each elemnt in Q_dagger
        for row in range(2**n_qb):
            for col in range(2**n_qb):
                i = n_qb
                while ((i > 1) and (Q_dagger_ifNonZero[row][col] != 0) ):
                    Q_dagger_ifNonZero_i, Q_dagger_sign_i, Q_dagger_frac_i = self.clebsch_gordan(J_M_chain[row][col][i-1], J_M_chain[row][col][i-2], [0.5, ms_chain[col][i-1] ])
                    if (Q_dagger_ifNonZero_i == 0):
                        Q_dagger_ifNonZero[row][col] = 0    # 1 for nonzero, 0 for zero
                        Q_dagger_sign[row][col] = 1         # 1 for plus, 0 for minus
                        Q_dagger_frac[row][col] = np.array([0, 1])
                    else:
                        if ((Q_dagger_sign_i == 1) and (Q_dagger_sign[row][col] == 1) ):
                            Q_dagger_sign[row][col] = 1
                        elif ((Q_dagger_sign_i == 0) and (Q_dagger_sign[row][col] == 0) ):
                            Q_dagger_sign[row][col] = 1
                        else:
                            Q_dagger_sign[row][col] = 0
                        Q_dagger_frac[row][col][0] = int(Q_dagger_frac[row][col][0] * Q_dagger_frac_i[0])   # 0 for numerator, 1 for denominator
                        Q_dagger_frac[row][col][1] = int(Q_dagger_frac[row][col][1] * Q_dagger_frac_i[1])
                        gcd = np.gcd(Q_dagger_frac[row][col][0], Q_dagger_frac[row][col][1])
                        Q_dagger_frac[row][col][0] = Q_dagger_frac[row][col][0] / gcd
                        Q_dagger_frac[row][col][1] = Q_dagger_frac[row][col][1] / gcd
                    i = i - 1
        
        # Calculate for numerical values of Q_dagger:
        for row in range(2**n_qb):
            for col in range(2**n_qb):
                if (Q_dagger_ifNonZero[row][col] == 0):
                    Q_dagger[row][col] = 0.0
                else:
                    if (Q_dagger_sign[row][col] == 1):
                        Q_dagger[row][col] = np.sqrt(Q_dagger_frac[row][col][0] / Q_dagger_frac[row][col][1] )
                    else:
                        Q_dagger[row][col] = - np.sqrt(Q_dagger_frac[row][col][0] / Q_dagger_frac[row][col][1] )
        
        Q = np.transpose(Q_dagger)

        J_subspace = J[-1]       #the value of J of all subspaces
        J_eigenstate = np.zeros((2**n_qb), dtype=np.float64)      #the value of J_n of all eigenstates
        M_eigenstate = np.zeros((2**n_qb), dtype=np.float64)      #the value of M_n of all eigenstates
        for i in range(2**n_qb):
            J_eigenstate[i] = J_chain[i][-1]
            M_eigenstate[i] = M[-1][i]

        return Q, Q_dagger, J_subspace, J_eigenstate, M_eigenstate
    
    def keep_original_H(self, Q_dagger_full, n_qb):
    # keep original H; Q is identity
        dim_H = 2**n_qb
        n_subdiag = 2**(n_qb - 1)
        Q_dagger = np.identity((dim_H), dtype=np.float64)
        Q = np.identity((dim_H), dtype=np.float64)
        
        return Q, Q_dagger, dim_H, n_subdiag
    
    def keep_original_Q(self, Q_dagger_full, n_qb):
    # keep original Q
        dim_H = 2**n_qb
        n_subdiag = 2**(n_qb) - 1
        Q_dagger = Q_dagger_full
        Q = np.transpose(Q_dagger)
        
        return Q, Q_dagger, dim_H, n_subdiag
    
    def solve_for_ECA_Q_Sn(self, Q_dagger_full, J_subspace, init_state, target_state, n_qb): 
    # Essential component analysis, reducing H by selecting essential subspaces of certain J's
    # Only works when there is no coupling term or there is all-to-all coupling term
        init_coeff = np.matmul(Q_dagger_full, init_state)      # coefficents of initial state when represented with |J, M> basis
        target_coeff = np.matmul(Q_dagger_full, target_state)     # coefficents of target state when represented with |J, M> basis
        init_eigenstate_sign = np.zeros((2**n_qb), dtype=np.int8)    # if the coefficient is zero or not
        target_eigenstate_sign = np.zeros((2**n_qb), dtype=np.int8)
        
        init_coeff = np.abs(init_coeff)
        target_coeff = np.abs(target_coeff)
        
        for i in range(2**n_qb):
            if (init_coeff[i] > 1e-8):
                init_eigenstate_sign[i] = 1        # 1 or non-zero, 0 for zero
            if (target_coeff[i] > 1e-8):
                target_eigenstate_sign[i] = 1
        
        dim_subspace_list = np.int64(2 * J_subspace + 1)     # dimension of subspaces, which is 2 * J + 1
        num_subspace = len(dim_subspace_list)                # number of subspaces
        init_subspace_sign = np.zeros((num_subspace), dtype=np.int8)    # if this subspace is involved or not
        target_subspace_sign = np.zeros((num_subspace), dtype=np.int8)
        sum_dim_subspace = 0                                 # sum of dimensions of all essential subspaces till the one that is being searched
        idx_subspace = 0
        for dim_subspace in dim_subspace_list:
            for i in range(dim_subspace):
                if (init_eigenstate_sign[sum_dim_subspace + i] == 1):
                    init_subspace_sign[idx_subspace] = 1
                if (target_eigenstate_sign[sum_dim_subspace + i] == 1):
                    target_subspace_sign[idx_subspace] = 1
            sum_dim_subspace = sum_dim_subspace + dim_subspace
            idx_subspace = idx_subspace + 1
        
        subspace_sign = np.logical_or(init_subspace_sign, target_subspace_sign)
        subspace_sign = np.multiply(subspace_sign, 1)
        dim_H = 0
        for i in range(n_qb):
            if (subspace_sign[i] == 1):
                dim_H = dim_H + dim_subspace_list[i]           #dimension of Hamiltonian
        
        # reduced Q_dagger consisting of essential J-subspace's
        Q_dagger = np.zeros((dim_H, 2**n_qb), dtype=np.float64)
        row_reduced = 0
        row_full = 0
        for i in range(n_qb):
            if (subspace_sign[i] == 1):
                for idx in range(dim_subspace_list[i]):
                    Q_dagger[row_reduced + idx] = Q_dagger_full[row_full + idx]
                row_reduced = row_reduced + dim_subspace_list[i]
            row_full = row_full + dim_subspace_list[i]
        
        Q = np.transpose(Q_dagger)
        
        n_subdiag = 1
        
        return Q, Q_dagger, dim_H, n_subdiag
    
    def solve_for_ECA_Q_Dn(self, Q_dagger_full, dimension_subspace, init_state, target_state, n_qb): 
    # Essential component analysis, reducing H by selecting essential subspaces
    # Only works for Dn symmetry
        init_coeff = np.matmul(Q_dagger_full, init_state)            # coefficents of initial state when represented with |J, M> basis
        target_coeff = np.matmul(Q_dagger_full, target_state)        # coefficents of target state when represented with |J, M> basis
        init_eigenstate_sign = np.zeros((2**n_qb), dtype=np.int8)    # if the coefficient is zero or not
        target_eigenstate_sign = np.zeros((2**n_qb), dtype=np.int8)
        
        init_coeff = np.abs(init_coeff)
        target_coeff = np.abs(target_coeff)
        
        for i in range(2**n_qb):
            if (init_coeff[i] > 1e-8):
                init_eigenstate_sign[i] = 1        # 1 or non-zero, 0 for zero
            if (target_coeff[i] > 1e-8):
                target_eigenstate_sign[i] = 1
        
        dim_subspace_list = np.int64(dimension_subspace)     # dimension of subspaces
        num_subspace = len(dim_subspace_list)                # number of subspaces
        init_subspace_sign = np.zeros((num_subspace), dtype=np.int8)    # if this subspace is involved or not
        target_subspace_sign = np.zeros((num_subspace), dtype=np.int8)
        sum_dim_subspace = 0                                 # sum of dimensions of all essential subspaces till the one that is being searched
        idx_subspace = 0
        for dim_subspace in dim_subspace_list:
            for i in range(dim_subspace):
                if (init_eigenstate_sign[sum_dim_subspace + i] == 1):
                    init_subspace_sign[idx_subspace] = 1
                if (target_eigenstate_sign[sum_dim_subspace + i] == 1):
                    target_subspace_sign[idx_subspace] = 1
            sum_dim_subspace = sum_dim_subspace + dim_subspace
            idx_subspace = idx_subspace + 1
        
        subspace_sign = np.logical_or(init_subspace_sign, target_subspace_sign)
        subspace_sign = np.multiply(subspace_sign, 1)        
        dim_H = 0
        for i in range(n_qb):
            if (subspace_sign[i] == 1):
                dim_H = dim_H + dim_subspace_list[i]           # dimension of Hamiltonian
        
        # reduced Q_dagger consisting of essential J-subspace's
        Q_dagger = np.zeros((dim_H, 2**n_qb), dtype=np.float64)
        row_reduced = 0
        row_full = 0
        for i in range(n_qb):
            if (subspace_sign[i] == 1):
                for idx in range(dim_subspace_list[i]):
                    Q_dagger[row_reduced + idx] = Q_dagger_full[row_full + idx]
                row_reduced = row_reduced + dim_subspace_list[i]
            row_full = row_full + dim_subspace_list[i]
        
        Q = np.transpose(Q_dagger)
        
        return Q, Q_dagger, dim_H
    
    def customize_Q(self, Q_dagger_full, n_qb):
    # Customize dim_H, n_subdiag, Q_dagger with user defined indices
        dim_H = 7
        n_subdiag = 2
        
        Q_dagger = np.zeros((dim_H, 2**n_qb), dtype=np.float64)
        row_keep = np.array([0, 1, 11, 2, 15, 3, 4], dtype=np.int64)    #length must be dim_H
        row = 0
        for i in row_keep:
            Q_dagger[row] = Q_dagger_full[i]
            row = row + 1
        
        Q = np.transpose(Q_dagger)
        
        return Q, Q_dagger, dim_H, n_subdiag
    
    def rearrange_by_M_Q(self, Q_dagger_full, n_qb, M_eigenstate):
    # Rearrange Q by M-clustering with J-clustered Q 
    # n qubits has (n+1) M-subspace's       
        dim_H = 2**n_qb
        
        if (n_qb % 2) == 0:     #if n_qb is even
            n_subdiag = comb(n_qb, int(n_qb/2), exact=True)
        else:                   #if n_qb is odd
            n_subdiag = comb(n_qb, int((n_qb-1)/2), exact=True)
        
        Q_dagger = np.zeros((dim_H, 2**n_qb), dtype=np.float64)
        M_subspace = np.zeros((n_qb + 1), dtype=np.float64)       #record M of each M-subspace
        subspace_dim = np.zeros((n_qb + 1), dtype=np.int64)       #record dimension of each M-subspace
        M_maximum = n_qb / 2
        idx_state = 0
        idx_subspace = 0
        
        for M in np.arange(M_maximum, - (M_maximum + 0.5), -1, dtype=np.float64):
            M_subspace[idx_subspace] = M
            subspace_dim_M = 0
            for i in range(2**n_qb):
                if np.abs(M - M_eigenstate[i]) < 0.1:
                    Q_dagger[idx_state] = Q_dagger_full[i]
                    idx_state = idx_state + 1
                    subspace_dim_M = subspace_dim_M + 1
            subspace_dim[idx_subspace] = subspace_dim_M
            idx_subspace = idx_subspace + 1
            
        Q = np.transpose(Q_dagger)
        
        return Q, Q_dagger, dim_H, n_subdiag
    
    def solve_for_selection_mat(self, H_coupling, Q_full, Q_dagger_full, J_subspace, n_qb):
    # calculate for how the J-subspace's are coupled to each other
        H_coupling_transformed = np.matmul(np.matmul(Q_dagger_full, H_coupling), Q_full)
        
        # Calculate for whether an eigenstate is coupled to another with H_coupling:
        sel_mat_eigenstate = np.zeros((2**n_qb, 2**n_qb), dtype=np.int8)       
        for i in range(2**n_qb):
            for j in range(2**n_qb):
                if (np.abs(H_coupling_transformed[i][j]) > 1e-15):
                    sel_mat_eigenstate[i][j] = 1
                
        dim_subspace_list = np.int8(2 * J_subspace + 1)
        dim_sel_mat_subspace = len(dim_subspace_list)
        sel_mat_subspace = np.zeros((dim_sel_mat_subspace, dim_sel_mat_subspace), dtype=np.int8)
        M_notConserved_subspace = np.zeros((dim_sel_mat_subspace, dim_sel_mat_subspace), dtype=np.int8)
        
        # Calculate for whether a J-subspace is coupled to another with H_coupling:
        # Calculate for whether M is a good quantum number with H_coupling
        for i in range(dim_sel_mat_subspace):
            for j in range(dim_sel_mat_subspace):
                idx_row_subspace = 0
                idx_col_subspace = 0
                for idx in range(i):
                    idx_row_subspace = idx_row_subspace + dim_subspace_list[idx]
                for idx in range(j):
                    idx_col_subspace = idx_col_subspace + dim_subspace_list[idx]
                dim_subspace_row = dim_subspace_list[i]
                dim_subspace_col = dim_subspace_list[j]
                row_col_difference = (idx_row_subspace - idx_col_subspace) + int((dim_subspace_row - dim_subspace_col) / 2 )
                for row in range(idx_row_subspace, (idx_row_subspace + dim_subspace_row), 1):
                    for col in range(idx_col_subspace, (idx_col_subspace + dim_subspace_col), 1):
                        if (sel_mat_eigenstate[row][col] == 1):
                            sel_mat_subspace[i][j] = 1
                            if ((row - col) != row_col_difference):
                                M_notConserved_subspace[i][j] = 1
            
        return sel_mat_subspace, sel_mat_eigenstate, M_notConserved_subspace
    
    def transform_Hamiltonian(self, Q, Q_dagger, dim_H):
    # Calculate for transformed H0, Hx, Hy and main diagonal of propagator U
        n_qb = self.n_qb
        n_subdiag = self.n_subdiag
        
        # Calculate for Q_dagger_Hx_Q and zero all elements that should be zero
        Q_dagger_Hx_Q = np.matmul(np.matmul(Q_dagger, self.Hx), Q)     
        ceiling_Hess_Hx = int((n_qb/2) * (n_qb/2 + 1) + 1 )      
        for row in range(dim_H):
            for col in range(dim_H):
                if ((np.abs(row - col) > n_subdiag) or (np.abs(row - col) == 0) ):
                    Q_dagger_Hx_Q[row][col] = 0.0
                else:
                    for i in range(0, ceiling_Hess_Hx):
                        if (np.abs((Q_dagger_Hx_Q[row][col])**2 - i) < 0.1):
                            Q_dagger_Hx_Q[row][col] = np.sqrt(i)
                            break
        
        # Calculate for Q_dagger_Hy_Q based on Q_dagger_Hx_Q
        Q_dagger_Hy_Q = np.matmul(np.matmul(Q_dagger, self.Hy), Q)
        for row in range(dim_H):
            for col in range(dim_H):
                if (((row - col) >= 1) and ((row - col) <= n_subdiag) ):
                    Q_dagger_Hy_Q[row][col] = 1j * Q_dagger_Hx_Q[row][col]
                elif (((row - col) <= -1) and ((row - col) >= - n_subdiag) ):
                    Q_dagger_Hy_Q[row][col] = -1j * Q_dagger_Hx_Q[row][col]
                else:
                    Q_dagger_Hy_Q[row][col] = 0.0
                
        Q_dagger_H0_Q = np.matmul(np.matmul(Q_dagger, self.H0), Q)
        # We can compare values in Q_dagger_H0_Q_diag with H0_diag when there is no coupling term
        # H0_diag = self.H0_diag

        for row in range(dim_H):
            for col in range(dim_H):
                if (np.abs(Q_dagger_H0_Q[row][col]) < 1e-15):     #1e-15 is an empirical value
                    Q_dagger_H0_Q[row][col] = 0.0
                        
        Q_dagger_H0_Q_diag = np.diag(Q_dagger_H0_Q)
        
        mat_U_diag = self.idn_nqb_diag + 1j * (self.tau/2) * Q_dagger_H0_Q_diag
        
        # We take sub-diagonal of Hx and Hy when there is no coupling term
        # sub_diag = np.diag(Q_dagger_Hx_Q, k=-1)
        # upper_diag = 1j * (self.tau/2) * np.concatenate(([0.0], sub_diag ), axis=0)
        # lower_diag = 1j * (self.tau/2) * np.concatenate((sub_diag, [0.0] ), axis=0)
        
        return mat_U_diag, Q_dagger_H0_Q, Q_dagger_Hx_Q, Q_dagger_Hy_Q
    
    def solve_for_n_subdiag(self, dim_H, Q_dagger_Hx_Q):
    # Output n_subdiag for Q_dagger_Hx_Q:
        n_subdiag = 0
        for i in range(0, dim_H):
            for j in range(0, dim_H):
                if np.abs(Q_dagger_Hx_Q[i][j]) > 1e-8 :
                    if int(np.abs(i - j) ) > n_subdiag :
                        n_subdiag = int(np.abs(i - j) )
        
        return n_subdiag
    
    def clebsch_gordan(self, J_M, j1_m1, j2_m2):
    # Return analytical expression of Clebsch-Gordan coefficient < j1, j2, J, M | j1, m1, j2, m2 >
    # only for j2 = 1/2 cases
        J = J_M[0]
        M = J_M[1]
        j1 = j1_m1[0]
        m1 = j1_m1[1]
        j2 = j2_m2[0]
        m2 = j2_m2[1]
        
        if (np.abs(M) > J):       #return 0 if |M| > J
            cg_ifNonZero = 0
            cg_sign = 1
            cg_frac = np.array([0, 1], dtype=np.int64)
            return cg_ifNonZero, cg_sign, cg_frac
        
        if (np.abs(m1) > j1):
            cg_ifNonZero = 0
            cg_sign = 1
            cg_frac = np.array([0, 1], dtype=np.int64)
            return cg_ifNonZero, cg_sign, cg_frac
        
        if (np.abs(m2) > j2):
            cg_ifNonZero = 0
            cg_sign = 1
            cg_frac = np.array([0, 1], dtype=np.int64)
            return cg_ifNonZero, cg_sign, cg_frac
        
        if ((m1 + m2) != M):      #return 0 if M is not conserved
            cg_ifNonZero = 0
            cg_sign = 1
            cg_frac = np.array([0, 1], dtype=np.int64)
            return cg_ifNonZero, cg_sign, cg_frac
        
        if not(((j1 + j2) == J) or (np.abs(j1 - j2) == J) ):   #return 0 if J doesn't satisfy selection rule
            cg_ifNonZero = 0
            cg_sign = 1
            cg_frac = np.array([0, 1], dtype=np.int64)
            return cg_ifNonZero, cg_sign, cg_frac
        
        if (m2 == 0.5):
            if ((j1 + 0.5) == J):
                cg_sign = 1
                cg_frac = np.array([int(2 * (j1 + 0.5 + M)), int(4 * (j1 + 0.5))], dtype=np.int64)
            else:
                cg_sign = 0
                cg_frac = np.array([int(2 * (j1 + 0.5 - M)), int(4 * (j1 + 0.5))], dtype=np.int64)
        else:
            if ((j1 + 0.5) == J):
                cg_sign = 1
                cg_frac = np.array([int(2 * (j1 + 0.5 - M)), int(4 * (j1 + 0.5))], dtype=np.int64)
            else:
                cg_sign = 1
                cg_frac = np.array([int(2 * (j1 + 0.5 + M)), int(4 * (j1 + 0.5))], dtype=np.int64)
                
        cg_ifNonZero = 1
        
        return cg_ifNonZero, cg_sign, cg_frac
    
    def solve_for_H_Sn(self, n_qb, sigma, eval_complex):
    # Calculate for first-order Hx, Hy, or Hz
        idn = self.idn
        
        if n_qb == 1:
            H = sigma
        elif n_qb == 2:
            H = np.kron(sigma, idn) + np.kron(idn, sigma)
        else:
            flag = np.identity(n_qb, dtype=np.int8)
            if eval_complex == False :
                H = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
            else:
                H = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.complex128)
            for j in range(0, n_qb):
                flag_j = flag[j]
                if eval_complex == False :
                    element_for_H_j = np.zeros((n_qb, 2, 2), dtype=np.float64)
                else:
                    element_for_H_j = np.zeros((n_qb, 2, 2), dtype=np.complex128)
                for k in range(0, n_qb):
                    if flag_j[k] == 0:
                        element_for_H_j[k] = idn
                    else:
                        element_for_H_j[k] = sigma
                H_j = np.kron(element_for_H_j[0], element_for_H_j[1])
                for k in range(2, n_qb):
                    H_j = np.kron(H_j, element_for_H_j[k])
                H[j] = H_j
            H = np.sum(H, axis=0)   
            
        return H
    
    def solve_for_H_coupling_ring(self, n_qb, sigma, eval_complex):
    # Calculate for second order H_coupling, ring coupling
        idn = self.idn
                
        if n_qb == 1:
            if eval_complex == False :
                H_coupling = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
            else:
                H_coupling = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        elif n_qb == 2:
            H_coupling = np.kron(sigma, sigma)
        else:
            flag = np.identity(n_qb, dtype=np.int8)    #whether sigmaz or identity-2
            for j in range(0, n_qb-1):
                flag[j][j+1] = 1             # 1 for sigmaz, 0 for identity-2
            flag[n_qb-1, 0] = 1
            if eval_complex == False :
                H_coupling = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.float64)
            else:
                H_coupling = np.zeros((n_qb, 2**n_qb, 2**n_qb), dtype=np.complex128)
            for j in range(0, n_qb):
                flag_j = flag[j]
                if eval_complex == False :
                    element_for_H_coupling_j = np.zeros((n_qb, 2, 2), dtype=np.float64)
                else:
                    element_for_H_coupling_j = np.zeros((n_qb, 2, 2), dtype=np.complex128)
                for k in range(0, n_qb):
                    if flag_j[k] == 0:
                        element_for_H_coupling_j[k] = idn
                    else:
                        element_for_H_coupling_j[k] = sigma
                H_coupling_j = np.kron(element_for_H_coupling_j[0], element_for_H_coupling_j[1])
                for k in range(2, n_qb):
                    H_coupling_j = np.kron(H_coupling_j, element_for_H_coupling_j[k])
                H_coupling[j] = H_coupling_j
            H_coupling = np.sum(H_coupling, axis=0)
        
        return H_coupling
    
    def solve_for_H_coupling_chain(self, n_qb, sigma, eval_complex):
    # Calculate for second order H_coupling, chain coupling
        idn = self.idn
                
        if n_qb == 1:
           if eval_complex == False :
               H_coupling = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
           else:
               H_coupling = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        elif n_qb == 2:
            H_coupling = np.kron(sigma, sigma)
        else:
            flag = np.identity(n_qb, dtype=np.int8)
            for j in range(0, n_qb-1):
                flag[j][j+1] = 1
            #flag[n_qb-1, 0] = 1        #ignore last row to be chain coupling
            if eval_complex == False :
                H_coupling = np.zeros((n_qb-1, 2**n_qb, 2**n_qb), dtype=np.float64)     #n_qb-1
            else:
                H_coupling = np.zeros((n_qb-1, 2**n_qb, 2**n_qb), dtype=np.complex128)
            for j in range(0, n_qb-1):     #n_qb-1
                flag_j = flag[j]
                if eval_complex == False :
                    element_for_H_coupling_j = np.zeros((n_qb, 2, 2), dtype=np.float64)
                else:
                    element_for_H_coupling_j = np.zeros((n_qb, 2, 2), dtype=np.complex128)
                for k in range(0, n_qb):
                    if flag_j[k] == 0:
                        element_for_H_coupling_j[k] = idn
                    else:
                        element_for_H_coupling_j[k] = sigma
                H_coupling_j = np.kron(element_for_H_coupling_j[0], element_for_H_coupling_j[1])
                for k in range(2, n_qb):
                    H_coupling_j = np.kron(H_coupling_j, element_for_H_coupling_j[k])
                H_coupling[j] = H_coupling_j
            H_coupling = np.sum(H_coupling, axis=0)
        
        return H_coupling
    
    def solve_for_H_coupling_lattice(self, n_qb, dim_lattice, sigma, eval_complex):
    # Calculate for second order H_coupling, lattice coupling
    # n_qb must be equal to row_lattice * col_lattice
        idn = self.idn
        row_lattice = dim_lattice[0]        #number of rows
        col_lattice = dim_lattice[1]        #number of columns
        #side_length = int(np.sqrt(n_qb))     #only for square lattice
        num_coupling = int(row_lattice * (col_lattice - 1) + col_lattice * (row_lattice - 1) )
        flag = np.zeros((num_coupling, n_qb), dtype=np.int8)
        l = 0
        for j in range(0, row_lattice):
            for k in range(0, col_lattice):
                if j != int(row_lattice - 1):
                    flag[l][int(j * col_lattice + k)] = 1
                    flag[l][int((j+1) * col_lattice + k)] = 1
                    l = l+1
                if k != int(col_lattice - 1):
                    flag[l][int(j * col_lattice + k)] = 1
                    flag[l][int(j * col_lattice + k + 1)] = 1
                    l = l+1
        
        if eval_complex == False :
            H_coupling = np.zeros((num_coupling, 2**n_qb, 2**n_qb), dtype=np.float64)
        else:
            H_coupling = np.zeros((num_coupling, 2**n_qb, 2**n_qb), dtype=np.complex128)
        for j in range(0, num_coupling):
            flag_j = flag[j]
            if eval_complex == False :
                element_for_H_coupling_j = np.zeros((n_qb, 2, 2), dtype=np.float64)
            else:
                element_for_H_coupling_j = np.zeros((n_qb, 2, 2), dtype=np.complex128)
            for k in range(0, n_qb):
                if flag_j[k] == 0:
                    element_for_H_coupling_j[k] = idn
                else:
                    element_for_H_coupling_j[k] = sigma
            H_coupling_j = np.kron(element_for_H_coupling_j[0], element_for_H_coupling_j[1])
            for k in range(2, n_qb):
                H_coupling_j = np.kron(H_coupling_j, element_for_H_coupling_j[k])
            H_coupling[j] = H_coupling_j
        H_coupling = np.sum(H_coupling, axis=0)
        
        return H_coupling
    
    def extract_band(self, H, dim_H, n_subdiag, eval_Hy):
    # extract sub-diagonal bands of H0, Hx, Hy
        if eval_Hy == True:            #Hy is complex
            H_ub = np.zeros((n_subdiag, dim_H), dtype=np.complex128)
            H_lb = np.zeros((n_subdiag, dim_H), dtype=np.complex128)
        else:                           #H0 and Hx are real
            H_ub = np.zeros((n_subdiag, dim_H), dtype=np.float64)
            H_lb = np.zeros((n_subdiag, dim_H), dtype=np.float64)
        
        for j in range(0, n_subdiag):
            for k in range(0, (dim_H - j - 1) ):
                H_lb[j][k] = H[j+k+1][k]
        
        for j in range(0, n_subdiag):
            for k in range((n_subdiag - j), dim_H):
                H_ub[j][k] = H_lb[n_subdiag - j - 1][k - (n_subdiag - j)]
                
        if eval_Hy == True:
            H_ub = np.conjugate(H_ub)
            
        return H_ub, H_lb
    
    def solve_for_quantum_gate(self, n_qb, gate):
        if n_qb == 1:
            gate_tensor_product = gate
        else:
            gate_tensor_product = gate
            for i in range(1, n_qb):
                gate_tensor_product = np.kron(gate_tensor_product, gate)
        
        return gate_tensor_product
    
    def gen_state(self, n_qb, state_para):
    # Generate states with tensor products, cannot be entangled states
        spinup = self.spinup
        spindown = self.spindown
        
        if state_para[0] == 0:
            state = spinup             # 0 for spin-up
        else:
            state = spindown           # 1 for spin-down
        
        if n_qb == 1:
            state = state
        else:
            element_for_state = np.zeros((n_qb, 2), dtype=np.float64)
            for j in range(0, n_qb):
                if state_para[j] ==0:
                    element_for_state[j] = spinup
                else:
                    element_for_state[j] = spindown
            for j in range(1, n_qb):
                state = np.kron(state, element_for_state[j])
        return state    
    
    def solve_fft(self, Bf, range_f, norm):
    # solve for FFT, the power spectrum
        pow_spec = np.abs(fft(Bf, range_f)/norm)
        return pow_spec
    
    def solve_ifft(self, Bf_fft, range_f, norm):
    # solve for inverse FFT, the B-field
        Bf = ifft(Bf_fft, range_f) * norm
        return Bf
    
    def init_Bf(self, random_seed):
    # Initialize B-fields with random noise
        np.random.seed(random_seed)   # remove the number when not testing for timing
        N = self.N
        Fs = 1.0 / self.tau
        Bf = .0002 * np.random.uniform(-1.0, 1.0, (N-1) )
        fft_Bf = fft(Bf, N-1)
        for j in range(int(6.0 / Fs * (N-1) ), N-1-int(6.0 / Fs * (N-1) ) ):
            fft_Bf[j] = 0.0           # remove high frequency components
        Bf = ifft(fft_Bf, N-1)
        Bf = np.real(Bf)
        return Bf
    
    def init_Bf_2(self, random_seed):
    # Initialize B-fields with sinusoidal basis with random phase factor
        np.random.seed(random_seed)   # remove the number when not testing for timing
        N = self.N
        Fs = 1.0 / self.tau
        t_field = self.t_field
        resolution = Fs / (N-1)
        
        Bf = np.zeros((N-1), dtype=np.float64)
        for j in range(1, int(6.0 / resolution) ):
            omega = j * 2 * pi * resolution
            phase = np.random.uniform(low = -pi, high = pi )
            Bf = Bf + .0002 * np.sin(omega * t_field + phase)
        return Bf
    
    def plot(self, Bx, By, J, P, Bx_amp_list, By_amp_list, grad_Bx_amp_list, grad_By_amp_list, f_max):                
    # Plot    
        plt.figure()
        iterNum = np.arange(0, len(P))
        plt.plot(iterNum, P, '-xr', markersize=4, label = 'P')
        plt.plot(iterNum, J, '-ob', markersize=4, label = 'J')
        plt.xlabel('Iteration Number')
        plt.ylabel('Functional Value')
        plt.legend()
        
        plt.figure()
        iterNum = np.arange(0, len(Bx_amp_list))
        plt.plot(iterNum, Bx_amp_list, '-xr', markersize=4, label = 'Bx_amp')
        plt.plot(iterNum, By_amp_list, '-ob', markersize=4, label = 'By_amp')
        plt.xlabel('Iteration Number')
        plt.ylabel('Functional Value')
        plt.legend()
        
        plt.figure()
        iterNum = np.arange(0, len(grad_Bx_amp_list))
        plt.plot(iterNum, grad_Bx_amp_list, '-xr', markersize=4, label = 'grad_Bx_amp')
        plt.plot(iterNum, grad_By_amp_list, '-ob', markersize=4, label = 'grad_By_amp')
        plt.xlabel('Iteration Number')
        plt.ylabel('Functional Value')
        plt.legend()
        
        plt.figure()
        plt.plot(self.t_field[:500], Bx[:500], '-r')
        plt.plot(self.t_field[:500], By[:500], '-b')
        plt.xlabel('$t$ (a.u.)')
        plt.ylabel('$B-field(t)$ (a.u.)')
        
        N = self.N
        fft_Bx = self.solve_fft(Bx, N-1, N-1)
        fft_By = self.solve_fft(By, N-1, N-1)
        f = self.f
        idx = int(np.argwhere(f<f_max)[-1])
        plt.figure()
        plt.plot(f[:idx], fft_Bx[:idx], '-r')
        plt.plot(f[:idx], fft_By[:idx], '-b')
        plt.xlabel('$\omega$ (a.u.)')
        plt.ylabel('$|\epsilon(\omega)|$ (a.u.)')
        
        plt.show()
        
        return None
