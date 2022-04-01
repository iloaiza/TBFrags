#FUNCTIONS FOR INTERFACING WITH PYTHON, USED FOR PASSING FROM FERMION OPERATOR <-> TWO-BODY TENSOR 
#AND FOR CALCULATING EXPECTATION VALUES AND VARIANCES
ENV["PYCALL_JL_RUNTIME_PYTHON"] = PY_DIR
using PyCall
using SparseArrays

pushfirst!(PyVector(pyimport("sys")["path"]), pwd()*"/UTILS")
np = pyimport("numpy")
scipy = pyimport("scipy")
sympy = pyimport("sympy")
of = pyimport("openfermion")
ham = pyimport("ham_utils")
fermionic = pyimport("ferm_utils")
mp2 = pyimport("mp2_utils")
qbit = pyimport("qubit_utils")
antic = pyimport("anti_commuting")
car2lcu = pyimport("car2lcu_utils")

of_simplify(OP) = of.reverse_jordan_wigner(of.jordan_wigner(OP))

function obtain_hamiltonian(mol_name; basis="sto3g", ferm=true, geometry=1, n_elec=false)
	if n_elec == false
		return ham.get_system(mol_name,ferm=ferm,basis=basis,geometry=geometry)
	else
		return ham.get_system(mol_name,ferm=ferm,basis=basis,geometry=geometry,n_elec=true)
	end		
end

function obtain_tbt(mol_name; basis="sto3g", ferm=true, spin_orb=true, geometry=1, n_elec=false)
	if n_elec == false	
		h_ferm = obtain_hamiltonian(mol_name, basis=basis, ferm=ferm, geometry=geometry)
	else
		h_ferm, num_elecs = obtain_hamiltonian(mol_name, basis=basis, ferm=ferm, geometry=geometry, n_elec=true)
	end

	tbt = fermionic.get_chemist_tbt(h_ferm, spin_orb=spin_orb)

	if n_elec == false
		return tbt, h_ferm
	else
		return tbt, h_ferm, num_elecs
	end
end

function full_ham_tbt(mol_name; basis="sto3g", ferm=true, spin_orb=true, geometry=1, n_elec=false)
	if spin_orb == false
		error("Spin_orb marked as false, can't combine one-body tensor into two-body tensor. Try using obtain_SD instead")
	end

	if n_elec == false	
		h_ferm = obtain_hamiltonian(mol_name, basis=basis, ferm=ferm, geometry=geometry)
	else
		h_ferm, num_elecs = obtain_hamiltonian(mol_name, basis=basis, ferm=ferm, geometry=geometry, n_elec=true)
	end

	tbt = fermionic.get_chemist_tbt(h_ferm, spin_orb=spin_orb)

	# BUILD OBT USING CHEMIST CORRECTION, SMALL INACCURACIES APPEAR
    #obt = fermionic.get_obt(h_ferm, spin_orb=spin_orb)
	#obt += fermionic.get_chemist_obt_correction(h_ferm, spin_orb=spin_orb)

	# BUILD OBT FROM DIFFERENCE OF FERMIONIC HAMILTONIAN WITH TWO-BODY CHEMIST OPERATOR, INACCURACIES ARE A LOT SMALLER
	h1b = h_ferm - tbt_to_ferm(tbt, spin_orb)
    h1b = of_simplify(h1b)
	obt = fermionic.get_obt(h1b, spin_orb=spin_orb)

	tbt += obt_to_tbt(obt)

	#@show of_simplify(h_ferm - tbt_to_ferm(tbt,spin_orb))
	

	if n_elec == false
		return tbt, h_ferm
	else
		return tbt, h_ferm, num_elecs
	end
end

function obtain_SD(mol_name; basis="sto3g", ferm=true, spin_orb=true, geometry=1, n_elec=false)
	if n_elec == false	
		h_ferm = obtain_hamiltonian(mol_name, basis=basis, ferm=ferm, geometry=geometry)
	else
		h_ferm, num_elecs = obtain_hamiltonian(mol_name, basis=basis, ferm=ferm, geometry=geometry, n_elec=true)
	end

	tbt = fermionic.get_chemist_tbt(h_ferm, spin_orb=spin_orb)
	h1b = h_ferm - tbt_to_ferm(tbt, spin_orb)
    h1b = of_simplify(h1b)
	obt = fermionic.get_obt(h1b, spin_orb=spin_orb)
	
	#@show of_simplify(h_ferm - tbt_to_ferm(tbt,spin_orb) - obt_to_ferm(obt,spin_orb))

	if n_elec == false
		return (obt, tbt), h_ferm
	else
		return (obt, tbt), h_ferm, num_elecs
	end
end

function tbt_to_ferm(tbt :: Array, spin_orb; norm_ord = NORM_ORDERED)
	if norm_ord == true
		return of.normal_ordered(fermionic.get_ferm_op(tbt, spin_orb))
	else
		return fermionic.get_ferm_op(tbt, spin_orb)
	end
end

function obt_to_ferm(obt, spin_orb; norm_ord = NORM_ORDERED)
	#one body tensor to fermionic operator
	if norm_ord == true
		return of.normal_ordered(fermionic.get_ferm_op(obt, spin_orb))
	else
		return fermionic.get_ferm_op(obt, spin_orb)
	end
end

function tbt_to_ferm(tbt :: Tuple, spin_orb; norm_ord = NORM_ORDERED)
	return tbt_to_ferm(tbt[2], spin_orb, norm_ord=norm_ord) + obt_to_ferm(tbt[1], spin_orb, norm_ord=norm_ord)
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

function expectation_value(op::PyObject, psi, n=of.count_qubits(op); 
							frag_flavour=META.ff, u_flavour=META.uf, NORM = NORM_BRAKETS, REAL = true)
	#returns expectation value of openfermion operator
	#if REAL=true, returns only real part
	if NORM == true
		e_val = of.expectation(of.get_sparse_operator(of.normal_ordered(op), n_qubits=n), psi)
	else
		e_val = of.expectation(of.get_sparse_operator(op, n_qubits=n), psi)
	end

	if REAL == true
		return real_round(e_val)
	else
		return e_val
	end
end

function expectation_value(frag::fragment, psi, n=frag.n; frag_flavour=META.ff, u_flavour=META.uf, norm_ord = NORM_ORDERED)
	#returns expectation value of normalized fragment (i.e. if frag = cn Fn -> returns <Fn>)
	op = fragment_to_normalized_ferm(frag, frag_flavour = frag_flavour, u_flavour = u_flavour, norm_ord = norm_ord)
	return expectation_value(op, psi, n, frag_flavour = frag_flavour, u_flavour = u_flavour)
end

function expectation_value(H :: SparseMatrixCSC, psi; 
							frag_flavour=META.ff, u_flavour=META.uf, NORM = NORM_BRAKETS, REAL = true)
	e_val = (psi' * H * psi)[1]

	if REAL == true
		return real_round(e_val)
	else
		return e_val
	end
end

function variance_value(op::PyObject, psi, n=of.count_qubits(op); frag_flavour=META.ff, u_flavour=META.uf, neg_tol = neg_tol, NORM = NORM_BRAKETS)
	if NORM == false
		var_val = of.variance(of.get_sparse_operator(op, n_qubits=n), psi)
	else
		var_val = of.variance(of.get_sparse_operator(of.normal_ordered(op), n_qubits=n), psi)
	end

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

function get_wavefunction(h_ferm, wfs, num_elecs = 0)
	if wfs == "fci"
		e_fci, psi = of.get_ground_state(of.get_sparse_operator(h_ferm))
	elseif wfs == "hf"
		println("Obtaining Hartree-Fock wavefunction with $num_elecs electrons")
		psi = fermionic.get_openfermion_hf(of.count_qubits(h_ferm), num_elecs)
	end

	return psi
end

function rotate_wavefunction_from_exp_generator(psi, G, coeff)
	#return exp(1im*coeff*G) |psi>
	return expmv(1im*coeff, G, psi)
	#Gmat = sparse((of.get_sparse_operator(G, n_qubits=n)).todense())
	
	#return expmv(1im*coeff, Gmat, psi)
end

function rotate_wavefunction_from_exp_ah_generator(psi, G, coeff, n=of.count_qubits(G))
	#return exp(coeff*G) |psi> <psi| (exp(-coeff*G))
	Gmat = coeff .* (of.get_sparse_operator(G, n_qubits=n)).todense()
	expG = exp(Gmat)
	expH = expG'
	expH2 = exp(-Gmat)
	if expH != expH2
		@show sum(abs2.(expH-expH2))
	else
		println("All checks out boss ;)")
	end

	return scipy.sparse.csr_matrix(expG * psi.todense() * expG')
end


function obtain_ccsd(mol_name; geometry=1, basis="sto3g", spin_orb=true)

	of_mol = ham.get_mol(mol_name, geometry=geometry, basis=basis)
	ccsd = mp2.get_ccsd_op(of_mol)
	#Hccsd = ccsd + of.hermitian_conjugated(ccsd)
	Hccsd = ccsd - of.hermitian_conjugated(ccsd)
	#Hccsd = 1im .*(ccsd - of.hermitian_conjugated(ccsd))
	
	tbt = 1im .* fermionic.get_chemist_tbt(Hccsd, spin_orb=spin_orb)

	return tbt, 1im * Hccsd
end

function array_rounder(A,digits=10)
	#rounds and array up to #digits, removes repeated values
    C = round.(A,digits=digits)
    B = [C[1]]
    lenA = length(A)

    for i in 2:lenA 
	    if C[i] == B[end]
	     	#repeated value
	    else
		    push!(B,C[i])
	    end
    end
    return B
end

function anti_commuting_decomposition(H::PyObject)
	return antic.get_antic_group(H)
end

function ac_sorted_inversion(H::PyObject, tol=1e20)
	return antic.sorted_inversion_antic(H, tol=tol)
end