import openfermion as of
import pauli_symplectic_vector_utils as psvu
import tapering_utils as tu

def taper_H_qubit(Hq : of.QubitOperator):
   n_qubits = of.count_qubits(Hq)
   H_PVL = psvu.PauliVecList(n_qubits, Hq)
   stabilizers = H_PVL.gen_stabilizers()

   for idx, stabilizer in enumerate(stabilizers):
      exp_val = of.expectation(of.get_sparse_operator(stabilizer, n_qubits), wf)
      stabilizers[idx] = np.real_if_close(exp_val/np.abs(exp_val)) * stabilizer
   Hq, fixed_positions, unitaries = tu.taper_off_qubits(Hq, stabilizers, output_tapered_positions = True)

   for unitary in unitaries:
      gs = of.get_sparse_operator(unitary, n_qubits) * gs
   fixed_position_sorted_idx = np.argsort(np.array(fixed_positions))

   for idx, i in enumerate(fixed_position_sorted_idx):
      gs = tu.wf_removed_qubit(gs, fixed_positions[i] - idx, n_qubits)
      n_qubits -= 1

   print(of.expectation(of.get_sparse_operator(Hq, n_qubits), gs))

   return Hq
