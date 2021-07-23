#FUNCTIONS FOR INTERFACING WITH PYTHON, USED FOR PASSING FROM FERMION OPERATOR <-> TWO-BODY TENSOR
ENV["PYCALL_JL_RUNTIME_PYTHON"] = PY_DIR
using PyCall

pushfirst!(PyVector(pyimport("sys")["path"]), pwd())
#np = pyimport("numpy")
of = pyimport("openfermion")
ham = pyimport("ham_utils")
fermionic = pyimport("ferm_utils")

function obtain_hamiltonian(mol_name; basis="sto3g", ferm=true, geometry=1, n_elec=false)
	if n_elec == false
		return ham.get_system(mol_name,ferm=ferm,basis=basis,geometry=geometry)
	else
		return ham.get_system(mol_name,ferm=ferm,basis=basis,geometry=geometry,n_elec=true)
	end		
end

function obtain_tbt(mol_name; basis="sto3g", ferm=true, spin_orb=true, geometry=1)
	h_ferm = obtain_hamiltonian(mol_name, basis=basis, ferm=ferm, geometry=geometry)
	tbt = fermionic.get_chemist_tbt(h_ferm, spin_orb=spin_orb)

	return tbt, h_ferm
end

function tbt_to_ferm(tbt, spin_orb; norm_ord = NORM_ORDERED)
	if norm_ord == true
		return of.normal_ordered(fermionic.get_ferm_op(tbt, spin_orb))
	else
		return fermionic.get_ferm_op(tbt, spin_orb)
	end
end

function fragment_to_ferm(frag; frag_flavour=META.ff, u_flavour=META.uf, norm_ord = NORM_ORDERED)
	tbt = fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
	return tbt_to_ferm(tbt, frag.spin_orb, norm_ord = norm_ord)
end

function fragment_to_normalized_ferm(frag; frag_flavour=META.ff, u_flavour=META.uf, norm_ord = NORM_ORDERED)
	tbt = fragment_to_normalized_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
	return tbt_to_ferm(tbt, frag.spin_orb, norm_ord = norm_ord)
end

function real_round(x,tol=real_tol)
	if abs(imag(x))/abs(x) >= tol
		println("Warning, rounding x=$x to real, complex component below tolerance!")
		return abs(x)
	end

	return real(x)
end

function expectation_value(op::PyObject, psi, n=of.count_qubits(op); frag_flavour=META.ff, u_flavour=META.uf)
	e_val = of.expectation(of.get_sparse_operator(op, n_qubits=n), psi)

	return real_round(e_val)
end

function expectation_value(frag::fragment, psi, n=frag.n; frag_flavour=META.ff, u_flavour=META.uf, norm_ord = NORM_ORDERED)
	op = fragment_to_normalized_ferm(frag, frag_flavour = frag_flavour, u_flavour = u_flavour, norm_ord = norm_ord)
	return expectation_value(op, psi, n, frag_flavour = frag_flavour, u_flavour = u_flavour)
end

function variance_value(op::PyObject, psi, n=of.count_qubits(op); frag_flavour=META.ff, u_flavour=META.uf, neg_tol = neg_tol)
	var_val = of.variance(of.get_sparse_operator(op, n_qubits=n), psi)

	var_val = real_round(var_val)
	if -neg_tol <= var_val < 0
		var_val = 0
	end

	return var_val
end

function variance_value(frag::fragment, psi, n=frag.n; frag_flavour=META.ff, u_flavour=META.uf, neg_tol = neg_tol, norm_ord = NORM_ORDERED)
	op = fragment_to_ferm(frag, frag_flavour = frag_flavour, u_flavour = u_flavour, norm_ord = norm_ord)
	return variance_value(op, psi, n, frag_flavour=frag_flavour, u_flavour=u_flavour, neg_tol=neg_tol)
end

function get_wavefunction(h_ferm, wfs)
	if wfs == "fci"
		_, psi = of.get_ground_state(of.get_sparse_operator(h_ferm))
	else
		error("Trying to obtain wavefunction for wfs=$wfs, not implemented!")
	end

	return psi
end