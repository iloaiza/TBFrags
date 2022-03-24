using SparseArrays, Arpack, ExpmV

#calculates gradients using parameter shift rules
include("py_utils.jl")

function energy_expectation(h_ferm, psi_hf, FRAG_MATS, taus)
	num_f = length(FRAG_MATS)
	if length(taus) != num_f
		error("Fragments array's dimensions $num_f is not equal to taus parameters dimensions $(length(taus))")
	end

	global psi_old = psi_hf

	for i in 1:num_f
		global psi_old = rotate_wavefunction_from_exp_generator(psi_old, FRAG_MATS[i], taus[i])
	end

	E = expectation_value(h_ferm, psi_old)
	#@show E

	return E
end

function energy_partial(h_ferm, psi_hf, FRAG_MATS, taus, k)
	#return gradient w/r to τ_k
	if typeof(META.ff) == CGMFR
		#2 eigenvalue operators with Ω = 2, uses shift-rule
		taus_plus = copy(taus)
		taus_minus = copy(taus)
		taus_plus[k] += π/4
		taus_minus[k] += -π/4
		e_diff = energy_expectation(h_ferm, psi_hf, FRAG_MATS, taus_plus) - energy_expectation(h_ferm, psi_hf, FRAG_MATS, taus_minus)

		return e_diff
	else
		error("Trying to calculate energy_grad from shift-rule for fragment flavour $(META.ff), not implemented!")
	end
end

function energy_grad(h_ferm, psi_hf, FRAG_MATS, taus)
	n = length(FRAG_MATS)
	E_grads = zeros(n)

	for i in 1:n
		E_grads[i] = energy_partial(h_ferm, psi_hf, FRAG_MATS, taus, i)
	end

	return E_grads
end

function ferm_op_to_sparse(OP, n_qubits, transformation)
	if transformation == "bravyi_kitaev" || transformation == "bk"
		op_trans =  of.bravyi_kitaev(OP)
	elseif transformation == "jordan_wigner" || transformation == "jw"
		op_trans = of.jordan_wigner(OP)
	else
		error("Trying to tranfrorm operator with transformation $transformation, not implemented!")
	end

	return sparse(of.get_sparse_operator(op_trans, n_qubits=n_qubits).todense())
end

function frags_to_sparse(FRAGS, n_qubits, transformation=transformation)
	FRAG_OPS = [fragment_to_normalized_ferm(frag) for frag in FRAGS]	
	FRAG_MATS = [ferm_op_to_sparse(op, n_qubits, transformation) for op in FRAG_OPS]

	return FRAG_MATS
end

function of_hf_to_sparse(psi_hf, tol=1e-10)
	#Converts open_fermion hf |ψ><ψ| to |ψ>
	#println("Obtaining wavefunction as vector...")
	@time E,psi = eigs(sparse(psi_hf.todense()), nev=1)
	if abs(E[1]-1) > tol
		println("Warning, diagonalizing pure density matrix resulted in eig=$(E[1])! Using corresponding wavefunction as pure")
	end
	psi = psi[1:end,1]
	return psi
end

function vqe_routine(Hmat, n_q, psi, FRAG_MATS; tau0 = Float64[], grad = true, transformation = transformation)
	#converts h_ferm and psi_hf to sparse array and vector respectively, then performs VQE optimization
	n = length(FRAG_MATS)
	taus = zeros(Float64, n)

	tlen = length(tau0)
	for i in 1:tlen
		taus[i] = tau0[i]
	end
	if tlen != 0
		#println("Running vqe routine with initial conditions:")
		taus[tlen+1:end] .= 0
		#@show taus
	end

	function cost(x)
		return energy_expectation(Hmat, psi, FRAG_MATS, x)
	end

	return optimize(cost, taus, BFGS())

	function grad!(storage, x)
		storage = energy_grad(Hmat, psi, FRAG_MATS, x)
		return storage
	end

	if grad == true
		sol = optimize(cost, grad!, taus, BFGS())
	else
		sol = optimize(cost, taus)
	end

	println("VQE routine converged, final cost is $(sol.minimum))")

	return sol	
end

function VQE_post(FRAGS, h_ferm, num_elecs; transformation = transformation, amps_type=amps_type)
	n_qubits = of.count_qubits(h_ferm)
	if typeof(FRAGS[1]) == fragment
		FRAG_MATS = frags_to_sparse(FRAGS, n_qubits, transformation)
	else
		FRAG_MATS = [ferm_op_to_sparse(frag, n_qubits, transformation) for frag in FRAGS]
	end

	psi_hf = get_wavefunction(h_ferm, "hf", num_elecs)
	psi_fci = get_wavefunction(h_ferm, "fci", num_elecs)
	E_hf = expectation_value(h_ferm, psi_hf)
	E_fci = expectation_value(h_ferm, psi_fci)
	Ecorr = E_fci - E_hf
	E_vqe = Float64[]
	SOLS = []
	CORRvqe = Float64[]
	ηvqe = Float64[]
	println("Starting VQE optimization using $(amps_type) decomposition fragments as generators")
	global tau0 = Float64[]

	println("Building objects for fast calculations...")
	t00 = time()
	Hmat = sparse((of.get_sparse_operator(h_ferm)).todense())
	psi = of_hf_to_sparse(psi_hf)
	println("Finished construction after $(time() - t00) seconds, starting VQE cycles...")

	for nn in 1:length(FRAGS)
		println("Starting using $nn first fragments...")
		@time tau_sol = vqe_routine(Hmat, n_qubits, psi, FRAG_MATS[1:nn],tau0=tau0, transformation=transformation)
		push!(E_vqe, tau_sol.minimum)
		#println("Minimizer for $nn fragments is: $(tau_sol.minimizer)")
		global tau0 = tau_sol.minimizer
		push!(CORRvqe, E_vqe[end] - E_hf)
		push!(ηvqe, CORRvqe[end] / Ecorr)
		println("Correlation energy approximated by $(round(ηvqe[end]*100,digits=3))%, error is $(round((E_vqe[end]-E_fci)*1000,digits=4)) mHa")
		println()
	end
	println("Finished VQE optimization")

	@show ηvqe
	@show E_vqe .- E_fci

	return CORRvqe, ηvqe, E_hf, E_fci
end