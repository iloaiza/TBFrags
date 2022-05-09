from openfermion import QubitOperator, commutator
from ferm_utils import get_on_idx
import numpy as np
import scipy as sp
from scipy import sparse

def get_n_qubit(H : QubitOperator):
    '''
    Get the number of qubits in H
    '''
    n = 0
    for pw, _ in H.terms.items():
        for ps in pw:
            n = max(n, ps[0])
    return n + 1

def get_pw_matrix(pw, n):
    '''
    Return the corresponding pauli matrix of given pauli-word 
    '''
    mat = 1
    # pauli_list = [np.identity(2) for i in range(n)]
    pauli_list = [sparse.identity(2) for i in range(n)]
    for ps in pw:
        if ps[1] == 'Z':
            # pauli_list[ps[0]] = np.array([[1, 0], [0, -1]])
            pauli_list[ps[0]] = sparse.csr_matrix([[1, 0], [0, -1]])
        elif ps[1] == 'X':
            # pauli_list[ps[0]] = np.array([[0, 1], [1, 0]])
            pauli_list[ps[0]] = sparse.csr_matrix([[0, 1], [1, 0]])
        elif ps[1] == 'Y':
            # pauli_list[ps[0]] = np.array([[0, -1j], [1j, 0]])
            pauli_list[ps[0]] = sparse.csr_matrix([[0, -1j], [1j, 0]])
    
    for pauli in pauli_list:
        # mat = np.kron(pauli, mat)
        mat = sparse.kron(pauli, mat)
    return mat

def get_qubit_hf_expectation(hf, op, tiny=1e-7):
    '''
    Return the expectation value <hf|op|hf> of some hf vector. e.g. |001>
    '''
    def has_str(pw, string):
        '''
        Checking if pw contains string: 'X', 'Y', or 'Z'
        '''
        for ps in pw:
            if ps[1] == string:
                return True
        return False
    def get_phase(pw, hf):
        '''
        Return \pm 1 depending on pauli-word and hf 
        '''
        # TODO: Could be wrong
        hf = np.flip(hf)
        phase = 1
        for ps in pw:
            if hf[ps[0]] == 1:
                phase *= -1
        return phase
    e = 0
    for pw, val in op.terms.items():
        if not has_str(pw, 'X') and not has_str(pw, 'Y'):
            e += val * get_phase(pw, hf)
    assert abs(np.imag(e)) < tiny
    return np.real(e)

def qubit_braket(lwf, rwf, op):
    '''
    Perform the qubit operator on rwf and check lwf == rwf 
    ''' 
    def qubit_action(pw, wf):
        '''
        Acting qubit operator pw onto wf. 
        Return the resulting vector and phase
        Example: ((1, X), (0, X)) |00> = |11>, phase=1
        '''
        phase = 1
        wf = np.flip(wf)
        for ps in pw:
            if ps[1] == 'X':
                wf[ps[0]] = not wf[ps[0]]
            elif ps[1] == 'Y':
                if wf[ps[0]] == 0:
                    phase *= -1j 
                else:
                    phase *= 1j
                wf[ps[0]] = not wf[ps[0]]
            else:
                if wf[ps[0]] == 1:
                    phase *= -1
        return np.flip(wf), phase

    e = 0
    for pw, val in op.terms.items():
        cur_rwf, phase = qubit_action(pw, np.copy(rwf))
        if all(lwf == cur_rwf):
            e += phase * val 
    return e

def qubit_hfp_braket(hfp, H):
    '''
    Given Hermitian Qubit Operator H and hf (bk form) pair [(hf_i, coeff_i), ...]
    Obtain <H> 
    '''
    e = 0
    nhf = len(hfp)
    for i in range(nhf):
        for j in range(i, nhf):
            cur_val = np.conj(hfp[i][1]) * hfp[j][1] * qubit_braket(hfp[i][0], hfp[j][0], H)
            if i == j:
                e += cur_val
            else:
                e += 2 * np.real(cur_val)
    return e

def get_qubit_matrix(H:QubitOperator, n=None):
    '''
    Get the matrix form of the qubit operator
    TODO: discrepancy found at heisenberg.py, debug
    '''
    if n is None:
        n = get_n_qubit(H)
    size = 2**n
    # mat = np.zeros((size, size), np.complex64)
    mat = sparse.csr_matrix((size, size), dtype=np.complex64) 

    for pw, val in H.terms.items():
        curmat = get_pw_matrix(pw, n)
        mat += curmat * val
    return mat.todense()

def qubit_wise_commuting(a : QubitOperator, b : QubitOperator):
    '''
    Check if a and b are qubit-wise commuting.
    assume a and b have only one term
    '''
    ps_dict = {}

    pw, _ = a.terms.copy().popitem()

    for ps in pw:
        ps_dict[ps[0]] = ps[1]

    pw, _ = b.terms.copy().popitem()
    for ps in pw:
        if ps[0] in ps_dict:
            if ps[1] != ps_dict[ps[0]]:
                return False

    return True

def largest_first(commuting_graph_complement):
    '''
    Let C be the complement of a commuting graph
    Return a dictionary where key index over colors 
    and values are a list of indices [i, j, ....] whose C[i, j] = 0
    '''
    n = commuting_graph_complement.shape[0]

    rows = commuting_graph_complement.sum(axis=0)
    ind = np.argsort(rows)[::-1]
    m = commuting_graph_complement[ind,:][:,ind]
    colors = dict()
    c = np.zeros(n, dtype=int)
    k = 0 #color

    for i in range(n):
        neighbors = np.argwhere(m[i,:])
        colors_available = set(np.arange(1, k+1)) - set(c[[x[0] for x in neighbors]])
        term = ind[i]
        if not colors_available:
            k += 1
            c[i] = k
            colors[c[i]] = [term]
        else:
            c[i] = min(list(colors_available))
            colors[c[i]].append(term)

    return colors

def recursive_largest_first(commuting_graph_complement):
    """
    Color the graph using "recursive largest first" heuristics with the given adjacency matrix
    Returns a dictionary with keys as colors (just numbers),
    and values as BinaryHamiltonian's
    Produces better results than LF but is slower
    """
    def n_0(m, colored):
        m_colored = m[list(colored)]
        l = m_colored[-1]
        for i in range(len(m_colored)-1):
            l += m_colored[i]
        white_neighbors = np.argwhere(np.logical_not(l))
        return set([x[0] for x in white_neighbors]) - colored

    n = commuting_graph_complement.shape[0]
    colors = dict()
    c = np.zeros(n, dtype=int)
    # so, the preliminary work is done

    uncolored = set(np.arange(n))
    colored = set()
    k = 0
    while uncolored:
        decode = np.array(list(uncolored))
        k += 1
        m = commuting_graph_complement[:, decode][decode, :]
        v = np.argmax(m.sum(axis=1))
        colored_sub = {v}
        uncolored_sub = set(np.arange(len(decode))) - {v}
        n0 = n_0(m, colored_sub)#vertices that are not adjacent to any colored vertices
        n1 = uncolored_sub - n0
        while n0:
            m_uncolored = m[:,list(n1)][list(n0),:]
            v = list(n0)[np.argmax(m_uncolored.sum(axis=1))]
            colored_sub.add(v) #stable
            uncolored_sub -= {v} #stable
            n0 = n_0(m, colored_sub)
            n1 = uncolored_sub - n0 #stable
        indices = decode[list(colored_sub)]
        c[indices] = k  # stable
        colors[k] = [i for i in indices] # stable
        colored |= set(indices)
        uncolored = set(np.arange(n)) - colored
    return colors

def get_qwc_group(H : QubitOperator):
    '''
    Return a list of qubit-wise commuting fragments of H
    '''
    # Preparing all terms in H into a list
    qubit_ops = []
    for pw, val in H.terms.items():
        qubit_ops.append(QubitOperator(term=pw, coefficient=val))
    n = len(qubit_ops)

    # Making commutation matrix
    comm_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            comm_matrix[i, j] = qubit_wise_commuting(qubit_ops[i], qubit_ops[j])

    # Compute commuting fragments
    comm_matrix = np.identity(n) + comm_matrix + comm_matrix.T
    colors = recursive_largest_first(1 - comm_matrix)

    # Collect commuting fragments into a list of QubitOperators
    qwc_list = []
    qwc_list_idx = 0
    for key, indices in colors.items():
        qwc_list.append(QubitOperator.zero())
        for idx in indices:
            qwc_list[qwc_list_idx] += qubit_ops[idx]
        qwc_list_idx += 1
    return qwc_list

def get_fc_group(H:QubitOperator):
    '''
    Return a list of commuting fragments of H
    '''
    # Building list of operators
    pws = []
    vals = []
    for pw, val in H.terms.items():
        pws.append(QubitOperator(term=pw, coefficient=1))
        vals.append(val)

    # Building commuting matrix and find commuting set
    pnum = len(pws)
    comm_mat = np.zeros((pnum, pnum))
    for i in range(pnum):
        for j in range(i+1, pnum):
            if commutator(pws[i], pws[j]) == QubitOperator.zero():
                comm_mat[i, j] = 1
    comm_mat = np.identity(pnum) + comm_mat + comm_mat.T 
    colors = recursive_largest_first(1 - comm_mat)

    comm_list = [QubitOperator.zero() for i in range(len(colors))]
    for key, indices in colors.items():
        for idx in indices:
            comm_list[key - 1] += pws[idx] * vals[idx]
    return comm_list

def get_bk_transformation(n):
    '''
    Obtain the n x n bk transformation matrix given n modes
    '''
    def get_bk_transformation_helper(k):
        '''
        Obtain the 2^k x 2^k bk transformation matrix 
        '''
        if k == 0:
            return 1
        else:
            mat = np.zeros((2**k, 2**k))
            half = 2**(k-1)
            mat[0, half:] = 1
            prev = get_bk_transformation_helper(k-1)
            mat[:half, :half] = prev
            mat[half:, half:] = prev
            return mat 
    k = int(np.ceil(np.log2(n)))
    bk_mat = get_bk_transformation_helper(k)

    dim = 2**k 
    diff = dim - n
    return bk_mat[diff:, diff:]

def get_bk_vec(onvec):
    '''
    Return the corresponding bk vector given a onvec 
    '''
    n = len(onvec)
    bkmat = get_bk_transformation(n)
    bkonvec = (bkmat @ onvec) % 2
    return bkonvec

def get_full_bk_vec(onvec):
    '''
    Return the corresponding bk vector given a onvec 
    in full 2^n basis
    '''
    n = len(onvec)
    bkonvec = get_bk_vec(onvec)
    vec = np.zeros((2**n, 1))
    vec[get_on_idx(bkonvec)] = 1
    return vec

def hfp2qubp(hfp):
    '''
    Transform a fermionic hf pair [(hf_i, coeff_i), ...]
    into its qubit equivalent
    '''
    qubp = []
    for p in hfp:
        qubhf = get_bk_vec(p[0])
        qubp.append((qubhf, p[1]))
    return qubp

def qubit_gs_variance(gs, list, tiny=1e-8):
    '''
    Obtain variances from bk form of gs, which is in paired hf form
    '''
    varis = np.zeros(len(list))
    for i, term in enumerate(list):
        cur_fci_ev = qubit_hfp_braket(gs, term)
        cur_fci_var = qubit_hfp_braket(gs, term*term) - cur_fci_ev ** 2
        if abs(cur_fci_var) < tiny:
            cur_fci_var = 0
        if np.imag(cur_fci_var) < tiny:
            cur_fci_var = np.real(cur_fci_var)
        varis[i] = cur_fci_var
    return varis

#Pauli to symplectic vector and back functions
def pauli_word_to_binary_vector(pauli_word, n_qubits):
    if isinstance(pauli_word, QubitOperator):
        pauli_word_tuple = list(pauli_word.terms.keys())[0]
    else:
        pauli_word_tuple = pauli_word
    binary_vec = np.zeros(2*n_qubits, dtype='int')
    for sigma in pauli_word_tuple:
        if sigma[1] == 'X':
            binary_vec[sigma[0]] = 1
        elif sigma[1] == 'Z':
            binary_vec[n_qubits+sigma[0]]= 1
        elif sigma[1] == 'Y':
            binary_vec[sigma[0]] = 1
            binary_vec[n_qubits+sigma[0]]= 1
    return binary_vec

def binary_vector_to_pauli_word(binary_vec):
    qubit_op_str = ''
    n_qubits = len(binary_vec) // 2
    for i in range(n_qubits):
        if binary_vec[i] == 1 and binary_vec[n_qubits+i] == 0:
            qubit_op_str += 'X{} '.format(i)
        elif binary_vec[i] == 1 and binary_vec[n_qubits+i] == 1:
            qubit_op_str += 'Y{} '.format(i)
        elif binary_vec[i] == 0 and binary_vec[n_qubits+i] == 1:
            qubit_op_str += 'Z{} '.format(i)
    return QubitOperator(qubit_op_str)

def is_anticommuting(string1, string2):
    #check if two binary vectors (i.e. Pauli words) are anticommuting
    n = len(string1) // 2

    x1 = string1[:n]
    z1 = string1[n:]
    x2 = string2[:n]
    z2 = string2[n:]

    return sum([x1[c] * z2[c] + z1[c] * x2[c] for c in range(n)]) % 2