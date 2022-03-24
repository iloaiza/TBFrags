"""
Functions related to obtaining MP2 initial amplitudes 
"""
import numpy as np
from openfermion import FermionOperator, normal_ordered, hermitian_conjugated, bravyi_kitaev, get_fermion_operator

def get_spin_orbital_energies(orbital_energies):
    ''' 
    Get the list of orbital energies. Return alpha/beta spin-orbital energies. 

    Args:
        orbital_energies: N orbital energies [..., e_i, ...]
    
    Returns:
        spin_orbital_energies: 2N spin orbital energies. [..., e_i, e_i, ...]
    '''
    norb = len(orbital_energies)
    spin_orbital_energies = np.zeros(2*norb)
    for i in range(norb):
        spin_orbital_energies[2*i] = orbital_energies[i]
        spin_orbital_energies[2*i+1] = orbital_energies[i]
    return spin_orbital_energies

def get_full_two_body_integrals(mol):
    '''
    Given MolecularData with Hamiltonian H = one_body + 1/2 * h_{pqrs} a^+_p a^+_q a_r a_s 
    Return h_{pqrs} in a 4-mode tensor 

    Args:
        mol: MolecularData of the Hamiltonian of interest
    
    Returns:
        h_{pqrs}: The two-body coefficients (multiplied by 2). 
    '''
    H = get_fermion_operator(mol.get_molecular_hamiltonian())

    n_spinorb = 2 * mol.n_orbitals
    h_pqrs = np.zeros((n_spinorb, n_spinorb, n_spinorb, n_spinorb))
    for term, val in H.terms.items():
        if len(term) == 4:
            p, q, r, s = term[0][0], term[1][0], term[2][0], term[3][0], 
            h_pqrs[p, q, r, s] = 2*val
    return h_pqrs

def get_ccsd_from_mp2(orbital_energies, two_body_integrals, occ):
    '''
    Obtain the ccsd initial guesses from orbital energies & two-body integrals
    '''
    n_spinorb = len(orbital_energies)
    orbs = list(np.arange(0, n_spinorb))
    vir = [x for x in orbs if x not in occ] # Collect virtual orbitals 

    ccsd_amp = np.zeros((n_spinorb, n_spinorb, n_spinorb, n_spinorb))
    for j in range(len(occ)):
        for i in range(j+1, len(occ)):
            for b in range(len(vir)):
                for a in range(b+1, len(vir)): 
                    # Following the formula. See theory/1701.02691.png or the paper
                    i_orb, j_orb, a_orb, b_orb = occ[i], occ[j], vir[a], vir[b]
                    ccsd_amp[i_orb, j_orb, a_orb, b_orb] =  \
                        two_body_integrals[i_orb, j_orb, b_orb, a_orb] - \
                        two_body_integrals[i_orb, j_orb, a_orb, b_orb] 
                    ccsd_amp[i_orb, j_orb, a_orb, b_orb] /= \
                        orbital_energies[i_orb] + orbital_energies[j_orb] -\
                        orbital_energies[a_orb] - orbital_energies[b_orb]
    return -ccsd_amp

def get_ccsd_amp(mol):
    '''
    Obtain MP2 CCSD amplitude from MolecularData
    '''
    occ = list(np.arange(0, mol.n_electrons))
    spin_orbital_energies = get_spin_orbital_energies(mol.orbital_energies)
    full_tbi = get_full_two_body_integrals(mol)
    ccsd_amp = get_ccsd_from_mp2(spin_orbital_energies, full_tbi, occ)
    ccsd_amp[abs(ccsd_amp) < 1e-10] = 0 # Zero low coefficient
    return ccsd_amp

def get_ccsd_op_from_amp(ccsd_amp):
    '''
    Obtain the ccsd operator from amplitudes 
    '''
    nzeros = np.nonzero(ccsd_amp) # in format ([1st indices], [2nd indices], ...)
    nind = len(nzeros[0])

    op = FermionOperator.zero()
    for curind in range(nind): # Going over each orbital, produce operator for each spin pair
        i, j, a, b = nzeros[0][curind], nzeros[1][curind], nzeros[2][curind], nzeros[3][curind]
        term = (
            (int(a), 1), (int(b), 1),
            (int(i), 0), (int(j), 0)
        )
        op += FermionOperator(term=term, coefficient=ccsd_amp[i, j, a, b])
        print((term,ccsd_amp[i,j,a,b]))
    return op

def get_ccsd_op(mol):
    '''
    Obtain the CCSD operator from MolecularData 
    '''
    ccsd_amp = get_ccsd_amp(mol)
    ccsd_op = get_ccsd_op_from_amp(ccsd_amp)
    return ccsd_op

def get_qubit_uccsd(mol):
    '''
    The top level function. Accepts a MolecularData object and compute BK UCCSD operator 
    based on MP2 amplitudes

    Args:
        mol: MolecularData of interesting Molecule
    
    Returns:
        uccsd_bk: Bravyi-Kitaev form of UCCSD operator 
    '''
    ccsd_op = get_ccsd_op(mol)
    uccsd_op = normal_ordered(ccsd_op + hermitian_conjugated(ccsd_op))
    return bravyi_kitaev(uccsd_op, mol.n_qubits)
