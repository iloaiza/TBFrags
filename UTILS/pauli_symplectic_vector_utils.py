"""Utility functions for openfermion's pauli-word manipulation. 
"""

from openfermion import QubitOperator
import openfermion as of
import qubit_utils as qubu
import numpy as np
import itertools

def check_ev_and_compute_if_missing(ev_dict, pw, wfs, n_qubits):
   key = get_pauli_word_tuple(pw)
   if not(key in ev_dict):
      pw_op = of.get_sparse_operator(get_pauli_word(pw), n_qubits)
      ev_dict[key] = np.real_if_close(of.expectation(pw_op, wfs)).item()
   return ev_dict

def get_pauli_word_tuple(P: QubitOperator):
    """Given a single pauli word P, extract the tuple representing the word. 
    """
    words = list(P.terms.keys())
    if len(words) != 1:
        raise(ValueError("P given is not a single pauli word"))
    return words[0]


def get_pauli_word(P: QubitOperator):
    """Given a single pauli word P, extract the same word with coefficient 1. 
    """
    words = list(P.terms.keys())
    if len(words) != 1:
        raise(ValueError("P given is not a single pauli word"))
    return QubitOperator(words[0])


def get_pauli_word_coefficient(P: QubitOperator, ghosts=None):
    """Given a single pauli word P, extract its coefficient. 
    """
    if ghosts is not None:
       if P in ghosts: 
          coeffs = [0.0]
       else:
          coeffs = list(P.terms.values())
    else:
       coeffs = list(P.terms.values())
    return coeffs[0]


def get_pauli_word_coefficients_size(P: QubitOperator):
    """Given a single pauli word P, extract the size of its coefficient. 
    """
    return np.abs(get_pauli_word_coefficient(P))


def get_pauliword_list(H: QubitOperator, ignore_identity=True):
    """Obtain a list of pauli words in H. 
    """
    pws = []
    for pw, val in H.terms.items():
        if ignore_identity:
            if len(pw) == 0:
                continue
        pws.append(QubitOperator(term=pw, coefficient=val))
    return pws


def qubit_wise_commuting(a: QubitOperator, b: QubitOperator):
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


def is_commuting(ipw, jpw, condition='fc'):
    """Check whether ipw and jpw are FC or QWC. 
    Args:
        ipw, jpw (QubitOperator): Single pauli-words to be checked for commutativity. 
        condition (str): "qwc" or "fc", indicates the type of commutativity to check. 

    Returns:
        is_commuting (bool): Whether ipw and jpw commute by the specified condition. 
    """
    if condition == 'fc':
        return of.commutator(get_pauli_word(ipw), get_pauli_word(jpw)) == QubitOperator.zero()
    else:
        return qubit_wise_commuting(ipw, jpw)

def get_pw_cov(ipw, jpw, ev_dict):
    """Obtain < ipw * jpw > - <ipw> * <jpw> given ev_dict. 
    Args:
        ipw, jpw (QubitOperator): Single pauli-words to be checked for commutativity. 
        ev_dict (Dict[tuple, float]): The dictionary where ev_dict[pw_tuple] = expectation(pw)
    
    Returns:
        prodev (float): The value < ipw * jpw >. < ipw * jpw > - <ipw> * <jpw> 
    """
    pdpw = ipw * jpw
    pdev = get_pauli_word_coefficient(pdpw) * ev_dict[get_pauli_word_tuple(pdpw)]
    ipw_ev = get_pauli_word_coefficient(ipw) * ev_dict[get_pauli_word_tuple(ipw)]
    jpw_ev = get_pauli_word_coefficient(jpw) * ev_dict[get_pauli_word_tuple(jpw)]
    
    return pdev - ipw_ev * jpw_ev
    

class PauliVecList():
   def __init__(self, *args, transpose=False):
      if len(args) >= 2: self.init_from_qo(args[0], list(args[1:]))
      else: self.init_from_array(args[0], transpose)

   def init_from_qo(self, nq, qos): 
      self.nq = nq
      qo = sum(qos)
      self.n_pvs = len(qo.terms)
      self.pvl = np.zeros((2*self.nq, self.n_pvs),dtype=np.int8)
      for ind_pv, pw in enumerate(qo.terms.keys()):
         for operator in pw:
            for i_q in range(self.nq):
               if operator[0] == i_q and operator[1] == 'X':
                  self.pvl[i_q, ind_pv] = 1 
               elif operator[0] == i_q and operator[1] == 'Y':
                   self.pvl[i_q, ind_pv] = 1
                   self.pvl[self.nq + i_q, ind_pv] = 1
               elif operator[0] == i_q and operator[1] == 'Z':
                   self.pvl[self.nq + i_q, ind_pv] = 1

   def init_from_array(self, input_array, transpose):
      if (transpose): self.pvl = np.array(input_array).transpose()
      else: self.pvl = np.array(input_array)

      if (self.pvl.dtype != np.int8): self.pvl = (self.pvl.astype(np.int8))

      array_shape = self.pvl.shape
      if (array_shape[0] % 2 == 0):
         self.nq = int(array_shape[0]/2)
         if (len(array_shape) == 1):
            self.n_pvs = 1
         else:
            self.n_pvs = int(array_shape[1])
      else:
         raise ValueError("Length of a Pauli vector has to be an even number.")
      if (self.pvl.dtype != np.int8): 
         raise ValueError("PauliVecList has to be a numpy array with dtype np.int8.")

   def __add__(self, pv2):
      self.check_compatibility(pv2, addition=True)
      return PauliVecList(self.pvl + pv2.pvl % 2)
         
   def __iadd__(self, pv2):
      self = self + pv2
      return self
         
   def __mul__(self, pv2):
      self.check_compatibility(pv2)
      J = np.block([[np.zeros((self.nq, self.nq)), np.eye(self.nq)], [np.eye(self.nq), np.zeros((self.nq, self.nq))]])
      if self.n_pvs == 1:
         return int(np.matmul(self.pvl, np.matmul(J, pv2.pvl))) % 2
      else:
         x = np.matmul(J, pv2.pvl)
         y = self.pvl.transpose()
         symplectic_prod = np.matmul(y, x)
         return symplectic_prod.astype(np.int8) % 2
         
   def __imul__(self, pv2):
      raise TypeError("Multiplication of two PauliVecLists does not yield a PauliVecList.")
      return self

   def __str__(self):
      s = self.pvl.transpose().__str__()
      return s

   def check_compatibility(self, pv2, addition=False):
      if (self.nq != pv2.nq): raise ValueError("Number of qubits of the two PauliVecLists has to match.")
      if (self.n_pvs != pv2.n_pvs and addition): raise ValueError("Number of PauliVectors of the two PauliVecLists has to match.")

   def new_PVL_commute_w_all(self, pv2):
      commute_w_all_arr_ind = self.commute_w_all(pv2)
      return PauliVecList(self.pvl[:, commute_w_all_arr_ind])

   def binary_null_space(self):
      dim = 2 * self.nq
      I = np.identity(dim, dtype = np.int8)
      binary_matrix = np.copy(self.pvl)
      for i_q in range(dim):
         non_zero_column = np.where(binary_matrix[i_q, :] == 1)
         if len(non_zero_column[0]) != 0:
            non_zero_column = non_zero_column[0][0]
            for j_q in range(i_q + 1, dim):
               if binary_matrix[j_q, non_zero_column] != 0:
                  binary_matrix[j_q, :] = (binary_matrix[i_q, :] + binary_matrix[j_q, :]) % 2
                  I[:, j_q] = (I[:, i_q] + I[:, j_q]) % 2
      where_zero = np.sum(binary_matrix, axis=1) == 0
      null_basis = I[:, where_zero]
      null_basis[:self.nq, :],null_basis[self.nq:, :] = null_basis[self.nq:, :].copy(),null_basis[:self.nq, :].copy()
      return null_basis

   def all_orthogonal_vectors(self):
      null_basis = self.binary_null_space()
      if not null_basis.all():
         N_all_vectors = 2 ** len(null_basis[0,:])
         all_vectors = np.zeros((len(null_basis[:,0]), N_all_vectors), dtype=np.int8)
         size_to_add = N_all_vectors // 2
         for i in range(len(null_basis[0,:])):
            for j in range( 2 ** i ):
               start = 2 * j * size_to_add
               for k in range(start, start + size_to_add):
                  all_vectors[:, k] += null_basis[:, i]
            size_to_add = size_to_add // 2
         return all_vectors[:, :-1] % 2
      else:
         return null_basis

   def pauli_operator(self, ind_pv):
      pauli_string = ''
      for i_q in range(self.nq):
         if (self.pvl[i_q, ind_pv] == 1 and self.pvl[self.nq + i_q, ind_pv] == 0):
            pauli_string += 'X' 
         if (self.pvl[i_q, ind_pv] == 1 and self.pvl[self.nq + i_q, ind_pv] == 1):
            pauli_string += 'Y' 
         if (self.pvl[i_q, ind_pv] == 0 and self.pvl[self.nq + i_q, ind_pv] == 1):
            pauli_string += 'Z' 
         if not(self.pvl[i_q, ind_pv] == 0 and self.pvl[self.nq + i_q, ind_pv] == 0):
            pauli_string += str(i_q)
            if (i_q != self.nq-1): pauli_string += ' '
      return pauli_string

   def pauli_operator_tuple(self, ind_pv):
      pauli_tuple = []
      for i_q in range(self.nq):
         if (self.pvl[i_q, ind_pv] == 1 and self.pvl[self.nq + i_q, ind_pv] == 0):
            pauli_tuple.append( (i_q, 'X') )
         elif (self.pvl[i_q, ind_pv] == 1 and self.pvl[self.nq + i_q, ind_pv] == 1):
            pauli_tuple.append( (i_q, 'Y') )
         elif (self.pvl[i_q, ind_pv] == 0 and self.pvl[self.nq + i_q, ind_pv] == 1):
            pauli_tuple.append( (i_q, 'Z') )
      return tuple(pauli_tuple)

   def gen_ghost_paulis(self, exclude=None, include=None):
      commuting_pvl = PauliVecList(self.all_orthogonal_vectors())
      if not (include == None):
         ghost_paulis_included = []
         dictionary = include.terms
         for i_pv in range(commuting_pvl.n_pvs):
            dict_key = commuting_pvl.pauli_operator_tuple(i_pv)
            if dict_key  in dictionary: 
               ghost_paulis_included.append( QubitOperator(term=dict_key, coefficient=dictionary[dict_key]) ) 
         return ghost_paulis_included
      else:
         ghost_paulis = []
         for i_pv in range(commuting_pvl.n_pvs):
            ghost_paulis.append( QubitOperator(term=commuting_pvl.pauli_operator(i_pv), coefficient=1.0) )
         if exclude == None:
            return ghost_paulis
         else:
            ghost_paulis_excluded = []
            for ghost in ghost_paulis:
               if not(qubu.get_pauli_word_tuple(ghost) in exclude.terms): ghost_paulis_excluded.append(ghost)
            return ghost_paulis_excluded

   def gen_stabilizers(self):
      commuting_pvl = PauliVecList(self.binary_null_space())
      stabilizers = []
      for i_pv in range(commuting_pvl.n_pvs):
         stabilizers.append( QubitOperator(term=commuting_pvl.pauli_operator(i_pv), coefficient = 1.0) )
      return stabilizers

def convert_a_to_xyz(ghost):
   replaced_ghost = []
   N_As = ghost.count('A')
   replaced_ghost.append(ghost.replace('A', 'X', 1))
   replaced_ghost.append(ghost.replace('A', 'Y', 1))
   replaced_ghost.append(ghost.replace('A', 'Z', 1))
   if N_As > 1:
      new_replaced_ghost = []
      for gst in replaced_ghost:
         new_replaced_ghost.extend(convert_a_to_xyz(gst))
      return new_replaced_ghost
   else:
      return replaced_ghost

class qwc_ghost_paulis():
   def __init__(self, *args):
      self.nq = args[0]
      qos = sum(list(args[1:]))
      qb_w_paulis = set()
      ghost_basis = []
      for key in qos.terms.keys():
         for pw in key:
            qb_w_paulis.add(int(pw[0]))
      for i in range(self.nq):
         if not (i in qb_w_paulis):
            ghost_basis.append('A'+str(i))
      self.ghosts = []
      if len(ghost_basis) > 0:
         N_all_ghosts = 2 ** len(ghost_basis)
         all_ghosts = [''] * N_all_ghosts
         size_to_multiply = N_all_ghosts // 2
         for i in range(len(ghost_basis)):
            for j in range( 2 ** i ):
               start = 2 * j * size_to_multiply
               for k in range(start, start + size_to_multiply):
                  all_ghosts[k] += ' ' + ghost_basis[i] 
                  all_ghosts[k] = all_ghosts[k].strip()
            size_to_multiply = size_to_multiply // 2
         for i in range(len(all_ghosts)-1):
            self.ghosts.extend(convert_a_to_xyz(all_ghosts[i]))
         self.convert_to_QubitOperator()

   def convert_to_QubitOperator(self):
      Qubit_operators = []
      for ghost in self.ghosts:
         Qubit_operators.append(of.QubitOperator(term=ghost, coefficient=1.0))
      self.ghosts = Qubit_operators

