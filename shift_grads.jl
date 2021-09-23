#calculates gradients using parameter shift rules
include("py_utils.jl")

function energy_expectation(h_ferm, psi_hf, FRAGS, taus)
	num_f = length(FRAGS)
	if length(taus) != num_f
		error("Fragments array's dimensions $num_f is not equal to taus parameters dimensions $(length(taus))")
	end

	global psi_old = psi_hf

	for i in num_f:-1:1
		op = fragment_to_normalized_ferm(FRAGS[i])
		psi_new = rotate_wavefunction_from_exp_generator(psi_old, op, taus[i])
		global psi_old = psi_new
	end

	E = expectation_value(h_ferm, psi_old)
	@show E

	return E
end

function energy_partial(h_ferm, psi_hf, FRAGS, taus, k)
	#return gradient w/r to τ_k
	if typeof(META.ff) == CGMFR
		#2 eigenvalue operators with Ω = 2
		taus_plus = copy(taus)
		taus_minus = copy(taus)
		taus_plus[k] += π/4
		taus_minus[k] += -π/4
		e_diff = energy_expectation(h_ferm, psi_hf, FRAGS, taus_plus) - energy_expectation(h_ferm, psi_hf, FRAGS, taus_minus)

		return e_diff
	else
		error("Trying to calculate energy_grad from shift-rule for fragment flavour $(META.ff), not implemented!")
	end
end

function energy_grad(h_ferm, psi_hf, FRAGS, taus)
	n = length(FRAGS)
	E_grads = zeros(n)

	for i in 1:n
		E_grads[i] = energy_partial(h_ferm, psi_hf, FRAGS, taus, i)
	end

	return E_grads
end

using SparseArrays, Arpack, ExpmV

function vqe_routine(h_ferm :: PyObject, psi_hf :: PyObject, FRAGS; tau0 = Float64[], grad = true)
	#converts h_ferm and psi_hf to sparse array and vector respectively, then performs VQE optimization
	n = length(FRAGS)
	taus = zeros(Float64, n)

	for i in 1:n
		taus[i] = FRAGS[i].cn[1]
	end

	tlen = length(tau0)
	for i in 1:tlen
		taus[i] = tau0[i]
	end
	if tlen != 0
		println("Running vqe routine with initial conditions:")
		taus[tlen+1:end] .= 0
		@show taus
	end

	Hmat = sparse((of.get_sparse_operator(h_ferm)).todense())
	
	println("Obtaining wavefunction as vector...")
	@time E,psi = eigs(sparse(psi_hf.todense()), nev=1)
	if E[1] != 1
		println("Warning, diagonalizing pure density matrix resulted in eig=$(E[1])! Using corresponding wavefunction as pure")
	end
	psi = psi[1:end,1]

	function cost(x)
		return energy_expectation(Hmat, psi, FRAGS, x)
	end

	return optimize(cost, taus, BFGS())

	function grad!(storage, x)
		storage = energy_grad(Hmat, psi, FRAGS, x)
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
