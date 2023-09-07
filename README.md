# qoc_multi-qubits

Copyright [2023] by Xian Wang

This project was developed to solve the optimal magnetic pulses B(t) exciting desired transitions in multi-qubit systems.
Input parameters in QOC_MultiQubits.py
Utilities and methods are in QOC_utils.py

In QOC_utils.py, choose which method to use in class QOC_qubits.

The following functions are for Lie-Trotter-Suzuki decomposition:
fwd_propagation_original()
fwd_propagation_transformed()
fwd_propagation_cpl_original()
fwd_propagation_cpl_transformed()
compare_two_propagation()
index_switch()
direct_product_single_qubit_S_n_minus_1()
direct_product_single_qubit_D_n_minus_1()
direct_product_two_spaces()

The following functions are for the gradient-based framework:
transition_prob()
golden_section_search()
optimize()

The following functions are for the Dn-symmetry-based method:
diag_operator_method_Dn()
diag_unitary_irrep_operator_Dn()
basis_for_diag_unitary_irrep_operator_Dn()
solve_for_Adjoint_Q_diag_method_Dn()

The following functions are for the character tables of Sn and Dn:
character_table()
character_table_Sn()
character_table_Dn()
Catalan_single()
Catalan_dual()
modified_Catalan()
integer_partition()

The following functions are for the Sn-symmetry-based method:
Young_method_Sn()
Young_tableau()
Weyl_tableau()
Young_operator()
row_col_2_young_op()
young_2_diag_op()
solve_for_Adjoint_Q_Young_method()
solve_for_Adjoint_Q_CG_method()
clebsch_gordan()

The following functions output the original and transformed H:
keep_original_H()
keep_original_Q()
solve_for_ECA_Q_Sn()
solve_for_ECA_Q_Dn()
customize_Q()
rearrange_by_M_Q()
solve_for_selection_mat()
transform_Hamiltonian()
solve_for_n_subdiag()

The following functions define the Hamiltonians, states and initial pulses:
solve_for_H_Sn()
solve_for_H_coupling_ring()
solve_for_H_coupling_chain()
solve_for_H_coupling_lattice()
extract_band()
solve_for_quantum_gate()
gen_state()
init_Bf()
init_Bf_2()

The following functions process and visualize the outputs:
solve_fft()
solve_ifft()
plot()
