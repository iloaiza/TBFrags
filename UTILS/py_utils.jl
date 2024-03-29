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
tap = pyimport("tapering_ham")

of_simplify(OP) = of.reverse_jordan_wigner(of.jordan_wigner(OP))

function qubit_transform(op, transformation=transformation)
	if transformation == "bravyi_kitaev" || transformation == "bk"
		op_qubit = of.bravyi_kitaev(op)
	elseif transformation == "jordan_wigner" || transformation == "jw"
		op_qubit = of.jordan_wigner(op)
	else
		error("$transformation not implemented for fermion to qubit operator maping")
	end

	return op_qubit
end

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

function ferm_to_tbt(op)
	#transform a 1+2 body fermionic operator into a single two-body tensor (in spin orbitals)
	op_norm = of.normal_ordered(op)
	tbt = fermionic.get_chemist_tbt(op_norm, spin_orb=true)
	h1b = op_norm - tbt_to_ferm(tbt, true)
    h1b = of_simplify(h1b)
	obt = fermionic.get_obt(h1b, spin_orb=true)
	tbt += obt_to_tbt(obt)

	#@show of_simplify(op - tbt_to_ferm(tbt, true))

	return tbt
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

function fragment_to_cartan_ferm(frag; frag_flavour=META.ff, norm_ord = NORM_ORDERED)
	#returns Cartan form of the tbt without applying unitary rotation in fragment
	tbt = fragment_to_normalized_cartan_tbt(frag, frag_flavour=frag_flavour)
	if CSA_family(frag_flavour) == false
		tbt = frag.cn[1] * tbt
	end
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
		println("Full CI energy is $e_fci")
	else
		error("Trying to obtain non-fci wavefunction, not defined for not qubit systems! If operator is in qubits already, try get_qubit_wavefunction instead.")
	end

	return psi
end

function get_qubit_wavefunction(h_qubit, wfs, num_elecs = 0)
	#returns wavefunction (i.e. not openfermion density matrix format) for Hamiltonian electronic state
	#use wfs="fci" of "hf" for full configuration interaction or hartree-fock wavefunction
	if wfs == "fci"
		e_fci, psi = of.get_ground_state(of.get_sparse_operator(h_qubit))
		println("Full CI energy is $e_fci")
	elseif wfs == "hf"
		println("Obtaining Hartree-Fock wavefunction with $num_elecs electrons, make sure qubit operator is in Jordan-Wigner form...")
		psi = fermionic.get_openfermion_hf(of.count_qubits(h_qubit), num_elecs)
		e_hf = of.expectation(of.get_sparse_operator(h_qubit), psi)
		println("Hartree-Fock energy is $(real(e_hf))")
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

function ac_sorted_insertion(H::PyObject, tol=1e20)
	return antic.sorted_insertion_antic(H, tol=tol)
end

function qubit_operator_trimmer(Qop, tol=1e-3)
	global H_trim = of.QubitOperator.zero()

	trim_count = 0
	trim_abs = 0.0
	for items in Qop.terms
		pw, val = items
		if abs(val) > tol
			global H_trim += of.QubitOperator(term=pw, coefficient=val)
		else
			trim_count += 1
			trim_abs += abs(val)
		end
	end
	
	println("Trimmed qubit operator of $trim_count words out of $(length(Qop.terms)), L1 norm of removed coefficients is $trim_abs")

	return H_trim
end

function of_wavefunction_to_vector(psi, tol=1e-8)
	psi_sparse = py_sparse_import(psi)
	@time E,psi = eigs(psi_sparse, nev=1, which=:LM)
    if abs(1-E[1]) > tol
        println("Warning, diagonalizing pure density matrix resulted in eig=$(E[1])! Using corresponding wavefunction as pure")
    end
    return psi[:,1]
end

function binary_is_anticommuting(bin1, bin2, n_qubits)
	#check if two binary vectors (i.e. Pauli words) are anticommuting

    return sum(bin1[1:n_qubits] .* bin2[n_qubits+1:end] + bin1[n_qubits+1:end] .* bin2[1:n_qubits]) % 2
end

function julia_ac_sorted_insertion(H::PyObject)
	#perform sorted insertion on QubitOperator from openfermion
	pws_orig, vals_orig = antic.get_nontrivial_paulis(H)

    pnum = length(pws_orig)

    Pauli_cost = sum(abs.(vals_orig))
    println("Pauli=$(Pauli_cost)($(ceil(log2(pnum)))), number of Paulis is $pnum")

    n_qubits = of.count_qubits(H)
    println("Allocating bin vectors")
    bin_vecs = zeros(Bool,2*n_qubits, pnum)

    println("Sorting by coefficients")
    ind_ord = sortperm(abs.(vals_orig))[end:-1:1]
    vals_ord = vals_orig[ind_ord]

    println("Filling binary vectors array")
    for i in 1:pnum
    	bin_vecs[:,i] = qbit.pauli_word_to_binary_vector(pws_orig[ind_ord[i]], n_qubits)
    end

    is_grouped = zeros(Bool,pnum)
    group_arrs = Array{Int64,1}[]
    vals_arrs = Array{Complex{Float64},1}[]

    println("Running sorted insertion algorithm")
    for i in 1:pnum
    	if is_grouped[i] == false
    		curr_group = [i]
    		curr_vals = [vals_ord[i]]
    		is_grouped[i] = true
    		for j in i+1:pnum
    			if is_grouped[j] == false
	    			if binary_is_anticommuting(bin_vecs[:,i],bin_vecs[:,j], n_qubits) == 1
	    				antic_w_group = true
	    				for k in curr_group[2:end]
	    					if binary_is_anticommuting(bin_vecs[:,k],bin_vecs[:,j], n_qubits) == 0
		    					antic_w_group = false
		    					break
		    				end
	    				end

	    				if antic_w_group == true
		    				push!(curr_group,j)
		    				push!(curr_vals,vals_ord[j])
		    				is_grouped[j] = true
		    			end
	    			end
	    		end
	    	end
    		push!(group_arrs,curr_group)
    		push!(vals_arrs, curr_vals)
    	end
    end

    if prod(is_grouped) == 0
    	println("Error, not all terms are grouped after AC-SI algorithm!")
    	@show is_grouped
    end

    num_groups = length(group_arrs)
    group_L1 = zeros(num_groups)
    for i in 1:num_groups
        for val in vals_arrs[i]
            group_L1[i] += abs2(val)
        end
    end

    L1_norm = sum(sqrt.(group_L1))
    return L1_norm, num_groups
end