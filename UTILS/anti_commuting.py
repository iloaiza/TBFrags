import numpy as np
from qubit_utils import recursive_largest_first
from ham_utils import get_system
from openfermion import bravyi_kitaev, get_sparse_operator, count_qubits, QubitOperator, commutator
import openfermion as of

def get_nontrivial_paulis(H:QubitOperator):
    pws = []
    vals = []

    for pw, val in H.terms.items():
        if pw != ():
            pws.append(QubitOperator(term=pw, coefficient=1))
            vals.append(val)

    return pws, vals


def get_antic_group(H:QubitOperator):
    '''
    Return a list of commuting fragments of H
    '''
    # Building list of operators
    pws, vals = get_nontrivial_paulis(H)
    #print("Found {} terms in Pauli representation".format(len(pws)))

    # Building commuting matrix and find commuting set
    pnum = len(pws)
    anticomm_mat = np.zeros((pnum, pnum))
    for i in range(pnum):
        for j in range(i+1, pnum):
            if commutator(pws[i], pws[j]) != QubitOperator.zero():
                anticomm_mat[i, j] = 1
    anticomm_mat = np.identity(pnum) + anticomm_mat + anticomm_mat.T 
    colors = recursive_largest_first(1 - anticomm_mat)

    group_L1 = np.zeros(len(colors))
    comm_list = [QubitOperator.zero() for i in range(len(colors))]
    for key, indices in colors.items():
        for idx in indices:
            comm_list[key - 1] += pws[idx] * vals[idx]
            group_L1[key - 1] += np.abs(vals[idx])**2
    #print("Number of different groups is {}".format(len(comm_list)))

    L1_norm = np.sum(group_L1**0.5)
    pauli_norm = np.sum(np.abs(vals))
    return comm_list, L1_norm, pauli_norm, len(pws)

def sorted_inversion_antic(H:QubitOperator, tol=1e-1):
    '''
    Return list of commuting fragments of H, found with sorted insertion algorithm
    Paulis with coefficients further away than tol are not grouped together
    '''
    pws_orig, vals_orig = get_nontrivial_paulis(H)

    pnum = len(pws_orig)
    vals_ord = np.flip(np.argsort(np.abs(vals_orig)))
    pws = []
    vals = []
    for i in range(pnum):
        pws.append(pws_orig[vals_ord[i]])
        vals.append(vals_orig[vals_ord[i]])
    
    grouped = np.zeros(pnum)

    groups_list = []
    op_list = []
    vals_arrs = []
    for i in range(pnum):
        if grouped[i] == 0:
            curr_group = [i]
            grouped[i] = 1
            curr_op = pws[i] * vals[i]
            curr_vals = [vals[i]]

            for j_ind in range(pnum-i-1):
                j = j_ind + i + 1
                if abs(vals[i]) - abs(vals[j]) < tol and commutator(pws[i], pws[j]) != QubitOperator.zero() and grouped[j] == 0:
                    antic_w_group = 0
                    for ind in curr_group:
                        if commutator(pws[j], pws[ind]) == QubitOperator.zero():
                            antic_w_group = 1
                            break
                    if antic_w_group == 0:
                        grouped[j] = 1
                        curr_op += pws[j]*vals[j]
                        curr_vals.append(vals[j])   
                        curr_group.append(j)             

            vals_arrs.append(curr_vals)
            op_list.append(curr_op)
            groups_list.append(curr_group)

    num_groups = len(op_list)
    group_L1 = np.zeros(num_groups)
    for i in range(num_groups):
        for val in vals_arrs[i]:
            group_L1[i] += np.abs(val)**2

    L1_norm = np.sum(group_L1**0.5)
    pauli_norm = np.sum(np.abs(vals))
    return op_list, L1_norm, pauli_norm, pnum, groups_list
