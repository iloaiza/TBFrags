function CSA_λs_evals(λs, n)
	E_VALS = zeros(typeof(λs[1,1]),2^n)
	λ_len = size(λs)[1]
	for i in 0:2^n-1
		n_arr = digits(i, base=2, pad=n)
		for j in 1:λ_len
			if n_arr[Int(λs[j,2])] == 1 && n_arr[Int(λs[j,3])] == 1
				E_VALS[i+1] += λs[j,1]
			end
		end
	end

	RANGE = [minimum(real.(E_VALS)), maximum(real.(E_VALS))]
	return RANGE
end

function CSA_obt_range(obt_CSA_so, tiny=1e-12)
	#Entry is a one-body spin-orbital Cartan tensor, returns operator range
	n = size(obt_CSA_so)[1]
	obt_diag = Diagonal(obt_CSA_so)

	if sum(abs.(obt_diag - obt_CSA_so)) > tiny
		println("Trying to calculate CSA tbt range for non CSA fragment, using op_range instead")
		return op_range(obt_to_ferm(obt_CSA_so, true))
	end

	obt_pos = (obt_diag + abs.(obt_diag))/2
	obt_neg = obt_diag - obt_pos

	return [sum(obt_neg), sum(obt_pos)]
end

function CSA_tbt_range(tbt_CSA_so, triang=false, tiny=1e-12)
	#Entry is a two-body spin-orbital Cartan tensor, returns operator range
	#if triang = false it assumes entry is symmetric (tbt_CSA_so[i,i,j,j] = tbt_CSA_so[j,j,i,i])
	#triang = true assumes all coefficients are on [i,i,j,j] for i≤j
	n = size(tbt_CSA_so)[1]
	tbt_diag = zeros(typeof(tbt_CSA_so[1]),n,n,n,n)
	for i in 1:n
		for j in 1:n
			tbt_diag[i,i,j,j] = tbt_CSA_so[i,i,j,j]
		end
	end

	if sum(abs.(tbt_diag - tbt_CSA_so)) > tiny
		println("Trying to calculate CSA tbt range for non CSA fragment, using op_range instead")
		return op_range(tbt_to_ferm(tbt_CSA_so, true))
	end

	if triang == false
		tbt_triang = cartan_tbt_to_triang(tbt_CSA_so)
	else
		tbt_triang = tbt_CSA_so
	end
	
	num_lambdas = Int(n*(n+1)/2)
	#λs[ij,1] = [λij]
	#λs[ij,2] = [ni num]
	#λs[ij,3] = [nj num]
	λs = zeros(typeof(tbt_CSA_so[1]),num_lambdas,3)
	global idx = 0
	for i in 1:n
		for j in i:n
			idx += 1
			λs[idx,1] = tbt_triang[i,i,j,j]
			λs[idx,2:3] = [i,j]
		end
	end

	return CSA_λs_evals(λs, n)
end

function py_sparse_import(py_sparse_mat; imag_tol=1e-14)
	#transform python sparse matrix into julia sparse matrix
	row, col, vals = scipy.sparse.find(py_sparse_mat)
	py_shape = py_sparse_mat.get_shape()
	n = py_shape[1]
	
	if sum(abs.(imag.(vals))) < imag_tol
		vals = real.(vals)
	end

	sparse_mat = sparse(row .+1, col .+1, vals, n, n)

	#@show sum(abs2.(sparse_mat - py_sparse_mat.todense()))

	return sparse_mat
end

function qubit_op_range(op_qubit, n_qubit=of.count_qubits(op_qubit); imag_tol=1e-16, ncv=minimum([50,2^n_qubit]), tol=1e-3)
	#Calculates maximum and minimum eigenvalues for qubit operator
	op_py_sparse_mat = of.qubit_operator_sparse(op_qubit)
	sparse_op = py_sparse_import(op_py_sparse_mat, imag_tol=imag_tol)

	if n_qubit >= 2
		E_max,_ = eigs(sparse_op, nev=1, which=:LR, maxiter = 500, tol=tol, ncv=ncv)
		E_min,_ = eigs(sparse_op, nev=1, which=:SR, maxiter = 500, tol=tol, ncv=ncv)
	else
		E,_ = eigen(collect(sparse_op))
		E = real.(E)
		E_max = maximum(E)
		E_min = minimum(E)
	end
	E_range = real.([E_min[1], E_max[1]])
	
	#= Debug, checks it's the same as full diagonalization
	E, _ = eigen(collect(sparse_op))
	E = real.(E)
	@show minimum(E)
	@show maximum(E)
	@show E_range - [minimum(E), maximum(E)]

	of_eigen = of.eigenspectrum(op_qubit)
	@show minimum(of_eigen)
	@show maximum(of_eigen)
	# =#
	
	
	return E_range
end

function op_range(op; transformation=transformation, imag_tol=1e-16)
	#Calculates maximum and minimum eigenvalues for fermionic operator
	op_qubit = qubit_transform(op, transformation)
	
	return qubit_op_range(op_qubit, imag_tol=imag_tol)
end

function cartan_to_qubit_naive_treatment(cartan_tbt, spin_orb)
	#input: cartan polynomial of n_i's
	#transforms fermionic cartan polynomial using 1-2ninj  
	#output: transforms into ∑λij/2 zi zj and returns sqrt norm
	q_op = of.QubitOperator.zero()
	tbt_so = tbt_to_so(cartan_tbt, spin_orb) / 2
	n = size(tbt_so)[1]

	for i in 1:n
		for j in 1:n
			if i != j
				zi = of.QubitOperator("Z$i")
				zj = of.QubitOperator("Z$j")
				zizj = of.QubitOperator("Z$i Z$j")
				q_op += 0.5*tbt_so[i,i,j,j] * (zi+zj-zizj)
			else
				z_string = "Z$i"
				q_op += tbt_so[i,i,i,i] * of.QubitOperator(z_string)
			end
		end
	end

	q_range = qubit_op_range(q_op, tol=1e-3)
	ΔE = (q_range[2] - q_range[1])/2

	#@show λVL1, λVL2, ΔE
	return ΔE
end

function cartan_to_qop(cartan_tbt, spin_orb, tol=1e-6)
	#input: cartan polynomial of n_i's
	#transforms fermionic tbt into (1-2ni)(1-2nj) -> zizj, requires correction to 1-body term
	#output: transforms into ∑λij/4 zi zj
	#also returns number of unitaries with norm ≥ tol
	q_op = of.QubitOperator.zero()
	tbt_so = tbt_to_so(cartan_tbt, spin_orb)
	n = size(tbt_so)[1]

	num_u = 0
	global λVL1 = 0.0
	for i in 1:n
		for j in 1:n
			if i != j
				global λVL1 += abs(tbt_so[i,i,j,j]) / 4
				z_string = "Z$(i-1) Z$(j-1)"
				q_op += (tbt_so[i,i,j,j] / 4) * of.QubitOperator(z_string)
				if tbt_so[i,i,j,j] / 4 ≥ tol
					num_u += 1
				end
			end
		end
	end

  	return q_op, λVL1, num_u
end

function cartan_to_qubit_l1_treatment(cartan_tbt, spin_orb)
	#input: cartan polynomial of n_i's
	#transforms fermionic tbt into (1-2ni)(1-2nj) -> zizj, requires correction to 1-body term
	#output: transforms into ∑λij/4 zi zj and returns:
	# L1 norm, sqrt norm, number of unitaries (tot, <tol)
  	q_op, λVL1, num_u = cartan_to_qop(cartan_tbt, spin_orb)
	q_range = qubit_op_range(q_op, tol=1e-3)
	ΔE = (q_range[2] - q_range[1])/2

	#λVL1/2 factor if using oblivious amplitude amplification
	return λVL1, ΔE, num_u
end

function L1_frags_treatment(CARTAN_TBTS, spin_orb, α_tot = size(CARTAN_TBTS)[1], n=size(CARTAN_TBTS)[2])
	#Calculate different L1 norms for group of fragments
	CSA_L1 = SharedArray(zeros(α_tot)) #holds (sum _ij |λij|) L1 norm for CSA polynomial
	E_RANGES = SharedArray(zeros(α_tot,2)) #holds eigenspectrum boundaries for each operator
	
	@sync @distributed for i in 1:α_tot
		#println("L1 treatment of fragment $i")
		#t00 = time()
		#build qubit operator for final sanity check
		
		CSA_L1[i] = cartan_tbt_l1_cost(CARTAN_TBTS[i,:,:,:,:], spin_orb)
		
		# SQRT_L1 subroutine
		op_CSA = tbt_to_ferm(CARTAN_TBTS[i,:,:,:,:], spin_orb)
		E_RANGES[i,:] = CSA_tbt_range(CARTAN_TBTS[i,:,:,:,:])
		
		#= linear programing reflection optimization
		obt_CSA_mo, tbt_CSA_mo = CARTAN_TBTS[i]
		tbt_CSA_mo = cartan_tbt_to_triang(tbt_CSA_mo)

		@time ref_sol = car2lcu.OBTTBT_to_L1opt_LCU(2obt_CSA_mo, 4tbt_CSA_mo, n, solmtd="l1ip", pout=false)
		@show CSA_L1_MO[i] = ref_sol["csa_l1"]
		@show CR_L1_MO[i] = ref_sol["lcu_l1"]	
		@show CR_FRAGS_MO[i] = ref_sol["poldim"]

		# =#
		# SPIN-ORBIT REFLECTIONS ROUTINE
		#=
		tbt_CSA_so = CARTAN_TBTS[i,:,:,:,:]

		tbt_triang = cartan_tbt_to_triang(tbt_CSA_so)
		#@show of_simplify(tbt_to_ferm(tbt_CSA_so, true) - tbt_to_ferm(tbt_triang, true))
		println("Starting full two-body, spin-orb optimization")
		@time ref_sol = car2lcu.TBT_to_L1opt_LCU(tbt_triang, n_qubit, solmtd="l1ip", pout=true)
		@show ref_sol["csa_l1"]
		# =#
		#println("Finished fragment $i after $(time() - t00) seconds...")
	end

	#=
	TBT_TOT = TBTS[1,:,:,:,:]
	for i in 2:α_tot
		TBT_TOT += TBTS[i,:,:,:,:]
	end
	# =#
	if spin_orb == false
		E_RANGES *= 4
	end

	return CSA_L1, E_RANGES
end

function qubit_treatment(H_q)
	#does quantum treatment of H_q qubit Hamiltonian
	#println("Starting AC-RLF decomposition")
	#@time op_list_AC, L1_AC, Pauli_cost, Pauli_num = anti_commuting_decomposition(H_q)
	
	#println("Starting AC-SI decomposition")
	#@time op_list, L1_sorted, Pauli_cost, Pauli_num,group_list = ac_sorted_insertion(H_q)
	#@show group_list
	#println("Pauli=$(Pauli_cost)($(ceil(log2(Pauli_num))))")
	#println("AC-RLF L1=$(L1_AC)($(ceil(log2(length(op_list_AC)))))")
	#println("AC-SI L1=$(L1_sorted)($(ceil(log2(length(op_list)))))")
	@time L1,num_ops = julia_ac_sorted_insertion(H_q)
	println("AC-SI L1=$(L1)($(ceil(log2(num_ops))))")
end

function H_POST(tbt, h_ferm, x0, K0, spin_orb; frag_flavour=META.ff, Q_TREAT=true, S2=true, linopt=true)
	#builds fragments from x0 and K0, and compares with full operator h_ferm and its associated tbt
	tbt_so = tbt_to_so(tbt, spin_orb)
	println("Obtaining symmetry-shifted Hamiltonian")
        if linopt == true
                #NOTE: S2=true is not finished. 
		@time tbt_ham_opt, x_opt = symmetry_linprog_optimization(tbt, spin_orb=true, S2=false)
	else
		@time tbt_ham_opt, x_opt = symmetry_cuadratic_optimization(tbt_so, true, S2=S2)
	end
	SVD_CARTAN_TBTS, SVD_TBTS, SVD_OP = tbt_svd(tbt_ham_opt, tol=1e-6, spin_orb=true)
	α_SVD = size(SVD_TBTS)[1]

	if typeof(tbt) <: Tuple
		n = size(tbt[1])[1]
	else
		n = size(tbt)[1]
	end

	if spin_orb == true
		n_qubit = n
	else
		n_qubit = 2n
	end
	
	#= CSA TREATMENT
	α_tot = length(K0)
	println("$α_tot total CSA fragments found")
	x_size = Int(length(x0)/α_tot)
	FRAGS = []
	TBTS = SharedArray(zeros(Complex{Float64}, α_tot, n_qubit, n_qubit, n_qubit, n_qubit))
	CARTAN_TBTS = SharedArray(zeros(Complex{Float64}, α_tot, n_qubit, n_qubit, n_qubit, n_qubit))
	PUR_CARTANS = SharedArray(zeros(Complex{Float64}, α_tot, n_qubit, n_qubit, n_qubit, n_qubit)) #purified Cartan tbts as two-body tensors in spin-orbitals
	PUR_COEFFS = SharedArray(zeros(α_tot, 5))

	fcl = frag_coeff_length(n, frag_flavour)
	println("Starting fragment building and purification for CSA...")
	@sync @distributed for i in 1:α_tot
		x_curr = x0[1+(i-1)*x_size:i*x_size]
		frag = fragment(x_curr[fcl+1:end], x_curr[1:fcl], K0[i], n_qubit, spin_orb)
		push!(FRAGS, frag)
		TBTS[i,:,:,:,:] = tbt_to_so(fragment_to_tbt(frag), spin_orb)
		CARTAN_TBTS[i,:,:,:,:] = tbt_to_so(fragment_to_normalized_cartan_tbt(frag), spin_orb)
		pur_tbt, pur_coeffs = cartan_tbt_purification(CARTAN_TBTS[i,:,:,:,:], true)
		PUR_CARTANS[i,:,:,:,:] = pur_tbt
		PUR_COEFFS[i,:] = pur_coeffs
	end
	# =#

	#= SVD purification treatment
	PUR_SVD_CARTANS = SharedArray(zeros(Complex{Float64}, α_SVD, n_qubit, n_qubit, n_qubit, n_qubit))
	PUR_SVD_COEFFS = SharedArray(zeros(Complex{Float64}, α_SVD, 5))
	PUR_SVD_TBTS = SharedArray(zeros(Complex{Float64}, α_SVD, n_qubit, n_qubit, n_qubit, n_qubit))
	@sync @distributed for i in 1:α_SVD
		println("Purifying Cartan fragment $i")
		orig_range =  CSA_tbt_range(SVD_CARTAN_TBTS[i,:,:,:,:])
		orig_l1 = cartan_tbt_l1_cost(SVD_CARTAN_TBTS[i,:,:,:,:], true)
		@time pur_tbt_svd, pur_coeffs_svd = cartan_tbt_purification(SVD_CARTAN_TBTS[i,:,:,:,:], true)
		pur_range = CSA_tbt_range(pur_tbt_svd)
		pur_l1 = cartan_tbt_l1_cost(pur_tbt_svd, true)
		println("Range of fragment $i modified from $orig_range to $pur_range")
		println("L1 of fragment $i modified from $orig_l1 to $pur_l1")
		PUR_SVD_CARTANS[i,:,:,:,:] = pur_tbt_svd
		PUR_SVD_COEFFS[i,:] = pur_coeffs_svd
		shift = shift_builder(pur_coeffs_svd, S_arr)
		PUR_SVD_TBTS[i,:,:,:,:] = SVD_TBTS[i,:,:,:,:] - shift
	end

	x = zeros(5)
	for i in 1:5
		x = PUR_SVD_COEFFS[:,i]
	end
	shift = shift_builder(x, S_arr)
	H_SYM_FERM_SVD = SVD_OP - tbt_to_ferm(shift, true)
	# =#

	S_arr = casimirs_builder(n_qubit, S2=true)

	println("Starting L1 treatment for SVD fragments")
	SVD_L1, SVD_E_RANGES = L1_frags_treatment(SVD_CARTAN_TBTS, true)
	#SVD_PUR_L1, SVD_PUR_RANGES = L1_frags_treatment(PUR_SVD_CARTANS, true)
	#= CSA subsection
	PUR_L1, PUR_RANGES= L1_frags_treatment(PUR_CARTANS, spin_orb)
	global H_SYM_FERM = of.FermionOperator.zero()
	for i in 1:α_tot
		x = PUR_COEFFS[i]
		shift = shift_builder(x, S_arr)
		global H_SYM_FERM += tbt_to_ferm(tbt_to_so(TBTS[i,:,:,:,:], spin_orb) - shift, true)
	end
	H_SYM_FERM = of_simplify(H_SYM_FERM)
	# =#

	println("Calculating range of full hamiltonian:")
	Etot_r = op_range(h_ferm)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2

	#= CSA
	println("Calculating range of symmetry shifted hamiltonian (CSA):")
	Etot_r = op_range(H_SYM_FERM, n_qubit)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	# =#

	#= SVD
	println("Calculating range of symmetry shifted hamiltonian (SVD):")
	Etot_r = op_range(H_SYM_FERM_SVD, n_qubit)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	# =#

	println("Showing range of symmetry-optimized hamiltonian")
	H_sym_opt = tbt_to_ferm(tbt_ham_opt, true)
	Etot_r = op_range(H_sym_opt)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	full_shift = shift_builder(x_opt, S_arr)
	@show of_simplify(h_ferm - H_sym_opt - tbt_to_ferm(full_shift, true))

	println("CSA L1 bounds (NR):")
	@show 
	#@show sum(CSA_L1)/2
	@show sum(SVD_L1)/2
	println("Shifted minimal norm (SR):")
	#ΔE_CSA = [(E_RANGES[i,2] - E_RANGES[i,1])/2 for i in 1:size(E_RANGES)[1]]
	#@show sum(ΔE_CSA)
	ΔE_SVD = [(SVD_E_RANGES[i,2] - SVD_E_RANGES[i,1])/2 for i in 1:α_SVD]
	@show sum(ΔE_SVD)
	#println("Purified L1 bounds (S-NR):")
	#@show sum(PUR_L1)/2
	#@show sum(SVD_PUR_L1)/2
	#println("Purified minimal norm (SS):")
	#ΔE_CSA_PUR = [(PUR_RANGES[i,2] - PUR_RANGES[i,1])/2 for i in 1:size(PUR_RANGES)[1]]
	#@show sum(ΔE_CSA_PUR)
	#ΔE_SVD_PUR = [(SVD_PUR_RANGES[i,2] - SVD_PUR_RANGES[i,1])/2 for i in 1:α_SVD]
	#@show sum(ΔE_SVD_PUR)

	if Q_TREAT == true
		println("Finished fermionic routine, starting qubit methods and final numbers...")
		H_full_q = qubit_transform(h_ferm)
		#H_sym_q = qubit_transform(H_SYM_FERM)
		#H_svd_q = qubit_transform(H_SYM_FERM_SVD)
		H_opt_q = qubit_transform(H_sym_opt)
		#H_tapered = tap.taper_H_qubit(H_full_q)
		#H_tapered_sym = tap.taper_H_qubit(H_sym_q)

		println("Full Hamiltonian:")
		qubit_treatment(H_full_q)

		#println("CSA shifted Hamiltonian:")
		#qubit_treatment(H_sym_q)

		#println("SVD shifted Hamiltonian:")
		#qubit_treatment(H_svd_q)

		println("Optimal shifted Hamiltonian:")
		qubit_treatment(H_opt_q)

		#exit()
		#sanity check, final operator recovers full Hamiltonian
		#= CSA sanity
		H_diff = h_ferm - CSA_OP
		H_qubit_diff = qubit_transform(H_diff)
		
		println("Difference from CSA fragments:")
		@show qubit_operator_trimmer(H_qubit_diff, 1e-5)
		# =#

		# = SVD sanity
		println("Difference from SVD fragments:")
		H_diff_SVD = h_ferm - SVD_OP - tbt_to_ferm(full_shift, true)
		H_qubit_diff_SVD = qubit_transform(H_diff_SVD)
		@show qubit_operator_trimmer(H_qubit_diff_SVD, 1e-3)

		#= Purified SVD sanity
		println("Difference from shifted SVD fragments:")
		x = zeros(5)
		for i in 1:5
			x[i] = sum(PUR_SVD_COEFFS[:,i])
		end
		shift = x[1]*Nα_tbt + x[2]*Nβ_tbt + x[3]*Nα2_tbt + x[4]*NαNβ_tbt + x[5]*Nβ2_tbt + x[6]*S2_tbt
		H_diff_SVD = h_ferm - H_SYM_FERM_SVD - tbt_to_ferm(shift, true)
		H_qubit_diff_SVD = qubit_transform(H_diff_SVD)
		@show qubit_operator_trimmer(H_qubit_diff_SVD, 1e-5)
		# =#
	end
end


function H_TREATMENT(tbt, h_ferm, spin_orb; Q_TREAT=true, S2=true, linopt=true)
	#builds fragments from x0 and K0, and compares with full operator h_ferm and its associated tbt
	tbt_so = tbt_to_so(tbt, spin_orb)
	println("Obtaining symmetry-shifted Hamiltonian")
        if linopt == true
                #NOTE: S2=true is not finished. 
		@time tbt_ham_opt, x_opt = symmetry_linprog_optimization(tbt, spin_orb=true, S2=false)
	else
		@time tbt_ham_opt, x_opt = symmetry_cuadratic_optimization(tbt_so, true, S2=S2)
		@show x_opt
	end
	SVD_CARTAN_TBTS, SVD_TBTS, SVD_OP = tbt_svd(tbt_so, tol=1e-6, spin_orb=true)
	PUR_SVD_CARTAN_TBTS, PUR_SVD_TBTS, PUR_SVD_OP = tbt_svd(tbt_ham_opt, tol=1e-6, spin_orb=true)

	α_SVD = size(SVD_TBTS)[1]
	PUR_α_SVD = size(PUR_SVD_TBTS)[1]

	@show of_simplify(tbt_to_ferm(tbt_ham_opt, true) - PUR_SVD_OP)
	@show of_simplify(tbt_to_ferm(tbt_so, true) - SVD_OP)

	n = size(tbt_so)[1]
	n_qubit = n
	
	S_arr = casimirs_builder(n_qubit, S2=S2)

	println("Starting L1 treatment for SVD fragments")
	@time SVD_L1, SVD_E_RANGES = L1_frags_treatment(SVD_CARTAN_TBTS, true)
	println("Starting L1 treatment for SVD fragments of purified Hamiltonian")
	@time PUR_SVD_L1, PUR_SVD_E_RANGES = L1_frags_treatment(PUR_SVD_CARTAN_TBTS, true)

	println("Starting purification and L1 treatment for SVD fragments:")
	SVD_THEN_PUR_CARTAN_TBTS = zeros(α_SVD, n_qubit, n_qubit, n_qubit, n_qubit)
	SVD_THEN_PUR_TBTS = zeros(α_SVD, n_qubit, n_qubit, n_qubit, n_qubit)
	X_OPTS = zeros(α_SVD, length(x_opt))
	TOT_SHIFT = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	for i in 1:α_SVD
		SVD_THEN_PUR_TBTS[i,:,:,:,:], X_OPTS[i,:] = symmetry_cuadratic_optimization(SVD_TBTS[i,:,:,:,:], true, S2=S2)
		shift = shift_builder(X_OPTS[i,:], S_arr)
		SVD_THEN_PUR_CARTAN_TBTS[i,:,:,:,:] = SVD_CARTAN_TBTS[i,:,:,:,:] - shift
		TOT_SHIFT += shift
	end
	if S2 == true
		println("Warning, Cartan norm will give wrong results for SVD-then-shifted since S² is not a Cartan form")
	end
	SVD_THEN_PUR_L1, SVD_THEN_PUR_E_RANGES = L1_frags_treatment(SVD_THEN_PUR_CARTAN_TBTS, true)

	println("Starting purification and L1 treatment for SVD fragments of symmetry-optimized H:")
	PUR2_CARTAN_TBTS = zeros(PUR_α_SVD, n_qubit, n_qubit, n_qubit, n_qubit)
	PUR2_TBTS = zeros(PUR_α_SVD, n_qubit, n_qubit, n_qubit, n_qubit)
	TOT_SHIFT_2 = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	X_OPTS_2 = zeros(PUR_α_SVD, length(x_opt))
	for i in 1:PUR_α_SVD
		PUR2_TBTS[i,:,:,:,:], X_OPTS_2[i,:] = symmetry_cuadratic_optimization(PUR_SVD_TBTS[i,:,:,:,:], true, S2=S2)
		shift = shift_builder(X_OPTS_2[i,:], S_arr)
		PUR2_CARTAN_TBTS[i,:,:,:,:] = PUR_SVD_CARTAN_TBTS[i,:,:,:,:] - shift
		TOT_SHIFT_2 += shift
	end
	if S2 == true
		println("Warning, Cartan norm will give wrong results for SVD-then-shifted since S² is not a Cartan form")
	end
	PUR2_L1, PUR2_E_RANGES = L1_frags_treatment(PUR2_CARTAN_TBTS, true)

	
	println("Calculating range of full hamiltonian:")
	@time Etot_r = op_range(h_ferm)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2

	println("Showing range of symmetry-optimized hamiltonian")
	H_sym_opt = tbt_to_ferm(tbt_ham_opt, true)
	@time Etot_r = op_range(H_sym_opt)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	full_shift = shift_builder(x_opt, S_arr)

	println("Showing range of SVD then symmetry-optimized hamiltonian")
	H_sym_opt_2 = tbt_to_ferm(tbt_so - TOT_SHIFT, true)
	@time Etot_r = op_range(H_sym_opt_2)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2

	println("Showing range of double symmetry-optimized hamiltonian")
	H_sym_opt_22 = tbt_to_ferm(tbt_ham_opt - TOT_SHIFT_2, true)
	@time Etot_r = op_range(H_sym_opt_22)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	
	println("CSA L1 bounds (NR):")
	@show sum(SVD_L1)/2
	@show sum(PUR_SVD_L1)/2
	@show sum(SVD_THEN_PUR_L1)/2
	@show sum(PUR2_L1)/2
	println("Fermionic L2 sums:")
	@show sum(SVD_L2)
	@show sum(PUR_SVD_L2)
	@show sum(SVD_THEN_PUR_L2)
	@show sum(PUR2_L2)
	println("Shifted minimal norm (SR):")
	ΔE_SVD = [(SVD_E_RANGES[i,2] - SVD_E_RANGES[i,1])/2 for i in 1:α_SVD]
	@show sum(ΔE_SVD)
	PUR_ΔE_SVD = [(PUR_SVD_E_RANGES[i,2] - PUR_SVD_E_RANGES[i,1])/2 for i in 1:PUR_α_SVD]
	@show sum(PUR_ΔE_SVD)
	SVD_THEN_PUR_ΔE = [(SVD_THEN_PUR_E_RANGES[i,2] - SVD_THEN_PUR_E_RANGES[i,1])/2 for i in 1:α_SVD]
	@show sum(SVD_THEN_PUR_ΔE)
	PUR2_ΔE = [(PUR2_E_RANGES[i,2] - PUR2_E_RANGES[i,1])/2 for i in 1:PUR_α_SVD]
	@show sum(PUR2_ΔE)

	if Q_TREAT == true
		println("Finished fermionic routine, starting qubit methods and final numbers...")
		H_full_q = qubit_transform(h_ferm)
		ψ_hf = get_qubit_wavefunction(H_full_q, "hf", num_elecs)
		println("Obtaining fci wavefunction for full Hamiltonian")
		@time ψ_fci = get_qubit_wavefunction(H_full_q, "fci", num_elecs)
		H_opt_q = qubit_transform(H_sym_opt)
		H_opt_q2 = qubit_transform(H_sym_opt_2)
		H_opt_q22 = qubit_transform(H_sym_opt_22)

		println("Full Hamiltonian:")
		qubit_treatment(H_full_q)

		println("Before shifted Hamiltonian:")
		qubit_treatment(H_opt_q)

		println("After shifted Hamiltonian:")
		qubit_treatment(H_opt_q2)

		println("Double shifted Hamiltonian:")
		qubit_treatment(H_opt_q22)

		exit()

		println("Tapering full Hamiltonian...")
		@time H_tapered = tap.taper_H_qubit(H_full_q, ψ_hf, ψ_hf)
		println("Showing ranges of full tapered Hamiltonian:")
		@time Etot_r = qubit_op_range(H_tapered)
		@show Etot_r
		@show (Etot_r[2] - Etot_r[1])/2
		println("Tapering symmetry-shifted Hamiltonian...")
		@time H_tapered_sym = tap.taper_H_qubit(H_opt_q, ψ_hf, ψ_hf)
		println("Showing ranges of symmetry-shifted tapered Hamiltonian:")
		@time Etot_r = qubit_op_range(H_tapered_sym)
		@show Etot_r
		@show (Etot_r[2] - Etot_r[1])/2
		println("Tapered Hamiltonian:")
		qubit_treatment(H_tapered)

		println("Tapered symmetry-shifted Hamiltonian:")
		qubit_treatment(H_tapered_sym)

		# = SVD sanity
		println("Difference from SVD fragments:")
		H_diff_SVD = h_ferm - SVD_OP
		H_qubit_diff_SVD = qubit_transform(H_diff_SVD)
		@show qubit_operator_trimmer(H_qubit_diff_SVD, 1e-5)

		println("Difference from purified SVD fragments:")
		H_diff_SVD_PUR = h_ferm - PUR_SVD_OP - tbt_to_ferm(full_shift, true)
		H_qubit_diff_SVD_PUR = qubit_transform(H_diff_SVD_PUR)
		@show qubit_operator_trimmer(H_qubit_diff_SVD_PUR, 1e-5)
	end
end

function QUBIT_TREATMENT(tbt, h_ferm, spin_orb; S2=true, linopt=true)
	#builds fragments from x0 and K0, and compares with full operator h_ferm and its associated tbt
	tbt_so = tbt_to_so(tbt, spin_orb)
	println("Obtaining symmetry-shifted Hamiltonian")
        if linopt == true
                #NOTE: S2=true is not finished. 
		@time tbt_ham_opt, x_opt = symmetry_linprog_optimization(tbt, spin_orb=true, S2=false)
	else
		@time tbt_ham_opt, x_opt = symmetry_cuadratic_optimization(tbt_so, true, S2=S2)
		@show x_opt
		#@time tbt_ham_opt_SD_S2, x_opt_SD_S2, u_params_S2 = orbital_mean_field_symmetry_reduction(tbt,spin_orb, u_flavour=MF_real(), S2=true, cartan=false)
		#@show x_opt_SD
	end
	@time tbt_ham_opt_SD, x_opt_SD, u_params = orbital_mean_field_symmetry_reduction(tbt, spin_orb, u_flavour=MF_real(), S2=false, cartan=false)
	#@show x_opt_SD_S2

	#@show tbt_cost(tbt_ham_opt_SD, 0)
	#@show tbt_cost(tbt_ham_opt_SD_S2, 0)

	#SVD_CARTAN_TBTS, SVD_TBTS, SVD_OP = tbt_svd(tbt_ham_opt_SD_S2, tol=1e-6, spin_orb=true)
	
	n = size(tbt_so)[1]
	n_qubit = n
	
	S_arr = casimirs_builder(n_qubit, S2=S2)

	
	#println("Showing range of symmetry-optimized hamiltonian")
	H_sym_opt = tbt_to_ferm(tbt_ham_opt, true)
	#@time Etot_r = op_range(H_sym_opt)
	#@show Etot_r
	#@show (Etot_r[2] - Etot_r[1])/2
	#full_shift = shift_builder(x_opt, S_arr)

	#=
	println("Showing range of unitary + symmetry-optimized hamiltonian")
	H_sym_opt_SD = tbt_to_ferm(tbt_ham_opt_SD, true)
	@time Etot_r = op_range(H_sym_opt_SD)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	#full_shift = shift_builder(x_opt, S_arr)
	# =# # =# # =# # =#

	println("Finished fermionic routine, starting qubit methods and final numbers...")
	H_full_q = qubit_transform(h_ferm)
	#ψ_hf = get_qubit_wavefunction(H_full_q, "hf", num_elecs)
	#println("Obtaining fci wavefunction for full Hamiltonian")
	#@time ψ_fci = get_qubit_wavefunction(H_full_q, "fci", num_elecs)
	H_opt_q = qubit_transform(H_sym_opt)
	#println("Tapering full Hamiltonian...")
	#@time H_tapered = tap.taper_H_qubit(H_full_q, ψ_hf, ψ_hf)
	#println("Showing ranges of full tapered Hamiltonian:")
	#@time Etot_r = qubit_op_range(H_tapered)
	#@show Etot_r
	#@show (Etot_r[2] - Etot_r[1])/2
	#println("Tapering symmetry-shifted Hamiltonian...")
	#@time H_tapered_sym = tap.taper_H_qubit(H_opt_q, ψ_hf, ψ_hf)
	#println("Showing ranges of symmetry-shifted tapered Hamiltonian:")
	#@time Etot_r = qubit_op_range(H_tapered_sym)
	#@show Etot_r
	#@show (Etot_r[2] - Etot_r[1])/2
	#=
	println("Tapering unitary + symmetry-shifted Hamiltonian...")
	H_opt_q_SD = qubit_transform(H_sym_opt_SD)
	@time H_tapered_sym_SD = tap.taper_H_qubit(H_opt_q_SD, ψ_hf, ψ_hf)
	println("Showing ranges of symmetry-shifted tapered Hamiltonian:")
	@time Etot_r = qubit_op_range(H_tapered_sym_SD)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	# =#


	println("Full Hamiltonian:")
	qubit_treatment(H_full_q)

	println("Shifted Hamiltonian:")
	qubit_treatment(H_opt_q)

	#println("Tapered Hamiltonian:")
	#qubit_treatment(H_tapered)

	#println("Tapered symmetry-shifted Hamiltonian:")
	#qubit_treatment(H_tapered_sym)

	#=
	println("Unitary + shifted Hamiltonian:")
	qubit_treatment(H_opt_q_SD)
	
	println("Tapered unitary + symmetry-shifted Hamiltonian:")
	qubit_treatment(H_tapered_sym_SD)
	# =#
end

function L1_ob_cost(obt_so)
	D,_ = eigen(obt_so)
	return 0.5 * sum(abs.(D))
end

function L1_ob_shift_cost(obt_so, t, Nα, Nβ)
	return L1_ob_cost(obt_so - t[1]*Nα - t[2]*Nβ)
end

function ob_L1_optimization(obt_so)
	n_so = size(obt_so)[1]
	OB_ARR,_ = casimirs_builder(n_so, one_body=true)
	Nα,Nβ = OB_ARR

	t0 = zeros(2)
	cost(x) = L1_ob_shift_cost(obt_so, x, Nα, Nβ)

	return optimize(cost, t0, BFGS())
end

function SHIFT_TREATMENT(tbt_mo_tup, h_ferm, ham_name)
	println("Starting qubit treatment...")
	# =
	println("Performing fermion to qubit mapping:")
	@time H_full_q = qubit_transform(h_ferm, "jw")
	#@show H_full_q
	println("\n\n\n Qubit treatment of Hamiltonian:")
	qubit_treatment(H_full_q)

	n_qubit = 2*size(tbt_mo_tup[1])[1]
	OB_ARR, TB_ARR = casimirs_builder(n_qubit, one_body=true)
	TB_Q_SYMS = TB_ARR[3:end]

	println("\n\n\n			STARTING SYMMETRY OPTIMIZATIONS ROUTINE: BEFORE")
	println("Calculations for 1+2 term")
	tbt_so = tbt_to_so(tbt_mo_tup, false)
	@time tbt_ham_opt, x_opt = symmetry_linprog_optimization(tbt_so, true)
	shift_op = tbt_to_ferm(tbt_ham_opt, true)
	shift_range = op_range(shift_op)
	@show (shift_range[2] - shift_range[1])/2
	println("Performing fermion to qubit mapping:")
	@time H_full_q = qubit_transform(shift_op, "jw")
	println("\n\n\n Qubit treatment of Hamiltonian:")
	qubit_treatment(H_full_q)
end

function MAJORANA_TREATMENT(tbt_mo_tup, h_ferm, ham_name, reps=10)
	println("Starting qubit treatment...")
	# =
	println("Performing fermion to qubit mapping:")
	@time H_full_q = qubit_transform(h_ferm, "jw")
	#@show H_full_q
	println("\n\n\n Qubit treatment of Hamiltonian:")
	qubit_treatment(H_full_q)

	println("Starting Majorana treatment...")
	@time maj_tup = majorana_parallel_U_optimizer(tbt_mo_tup, reps)
	H_maj = qubit_transform(tbt_to_ferm(maj_tup, false), "jw")
	println("\n\n\n Majorana qubit treatment of Hamiltonian:")
	qubit_treatment(H_maj)

	return 0
end

function QUBIT_L1(h_ferm,transform=transformation)
	# return L1 norm for Pauli and Anti-commuting grouping
	# input: fermionic Hamiltonian as openfermion FermionOperator
	println("Starting qubit L1 treatment routine")
	println("Performing fermion to qubit mapping:")
	@time H_full_q = qubit_transform(h_ferm, transform)
	println("\n\n\n Qubit treatment of Hamiltonian:")
	qubit_treatment(H_full_q)
end

function OUTDATED_TREATMENTS(tbt_mo_tup, h_ferm, ham_name)
	# MAJORANA ROUTINE
	println("Starting Majorana treatment...")
	@time maj_tup = majorana_parallel_U_optimizer(tbt_mo_tup, 40)
	H_maj = qubit_transform(tbt_to_ferm(maj_tup, false), "jw")
	println("\n\n\n Majorana qubit treatment of Hamiltonian:")
	qubit_treatment(H_maj)

	#T' BOOSTED AC ROUTINE
	println("Starting T' boosted AC-SI routine")
	@time L1,num_ops = bin_anticommuting_jw_sorted_insertion(tbt_mo_tup[1], tbt_mo_tup[2], cutoff = 1e-6)
	println("BOOSTED AC-SI L1=$(L1)($(ceil(log2(num_ops))))\n\n\n")


	SVD_tol = 1e-6
	CSA_tol = 1e-6

	println("\n\n\n Starting SVD routine for separated 1 and 2-body terms with cutoff tolerance $SVD_tol")
	@time NR_2B, SR_2B = svd_optimized(tbt_mo_tup[2], tol=SVD_tol, spin_orb=false)
	D,U = eigen(tbt_mo_tup[1])
	RANGES = zeros(1,2)
	obt_so = obt_orb_to_so(Diagonal(D))
	RANGES[:] = CSA_obt_range(obt_so)
	SR_2B = vcat(SR_2B, RANGES)
	push!(NR_2B, cartan_obt_l1_cost(D, false))
	ΔE_2B = [(SR_2B[i,2] - SR_2B[i,1])/2 for i in 1:length(NR_2B)]

	println("CSA L1 bounds (NR) (SVD 1-2):")
	@show sum(NR_2B)/2

	println("Shifted minimal norm (SR) (SVD 1-2):")
	@show sum(ΔE_2B)

	n_orbs = size(tbt_mo_tup[1])[1]
	n_qubit = 2*n_orbs
	OB_ARR, TB_ARR = casimirs_builder(n_qubit, one_body=true)
	TB_Q_SYMS = TB_ARR[3:end]
	SYM_MAT = τ_mat_builder(TB_Q_SYMS)


	println("\n\n Starting Reflection routine for separated 1 and 2-body terms")
	CGMFR_NAME = "CGMFR_" * ham_name
	FRAGS = run_optimization("g", tbt_mo_tup[2], CSA_tol, 1, 500, false, false, Float64[], Int64[], false, CGMFR_NAME, frag_flavour=CGMFR(), u_flavour=MF_real())
	obt_mod = tbt_mo_tup[1]
	tbt_tot = 0 .* tbt_mo_tup[2]
	global λCRT = 0
	for frag in FRAGS
		obt_curr, tbt_curr = fragment_to_tbt(frag, frag_flavour = CGMFR(), u_flavour = MF_real())
		obt_mod += obt_curr
		tbt_tot += tbt_curr 
		global λCRT += abs(frag.cn[1])
	end
	@show sum(abs.(tbt_tot - tbt_mo_tup[2]))
	obt_D, _ = eigen(obt_mod)
	λT = sum(abs.(obt_D))
	@show λT, λCRT, λT + λCRT


	println("\n\n Starting GT routine for separated 1 and 2-body terms")
	obt_mod = tbt_mo_tup[1]
	tbt_tot = copy(tbt_mo_tup[2])
	#for i in 1:n
	#	obt_mod[i,i] += tbt_tot[i,i,i,i]
	#	tbt_tot[i,i,i,i] = 0
	#end
	GT_NAME = "GT_" * ham_name
	FRAGS = run_optimization("g", tbt_tot, CSA_tol, 1, 500, false, false, Float64[], Int64[], false, GT_NAME, frag_flavour=G2(), u_flavour=MF_real())
	α_GT = length(FRAGS)
	global λGT = 0
	for frag in FRAGS
		obt_curr, tbt_curr = fragment_to_tbt(frag, frag_flavour = GT(), u_flavour = MF_real())
		obt_mod += obt_curr
		tbt_tot += tbt_curr 
		global λGT += abs(frag.cn[1])
	end
	@show sum(abs.(tbt_tot - tbt_mo_tup[2]))
	obt_D, _ = eigen(obt_mod)
	λT = sum(abs.(obt_D))
	@show λT, λGT, λT + λGT
end

function CSA_L1(tbt_mo_tup, h_ferm, ham_name, CSA_tol=1e-6)
	# DO L1 CALCULATION ROUTINE USING CSA DECOMPOSITION
	# INCLUDES BOTH WITHOUT AND WITH SYMMETRY OPTIMIZATION FOR INDIVIDUAL FRAGMENTS
	n_orbs = size(tbt_mo_tup[1])[1]
	n_qubit = 2*n_orbs
	OB_ARR, TB_ARR = casimirs_builder(n_qubit, one_body=true)
	TB_Q_SYMS = TB_ARR[3:end]
	SYM_MAT = τ_mat_builder(TB_Q_SYMS)

	println("\n\n Starting CSA routine for separated 1 and 2-body terms")
	CSA_2_NAME = "CSA_2_" * ham_name 
	n = size(tbt_mo_tup[1])[1]
	obt_tilde = tbt_mo_tup[1] + 2*sum([tbt_mo_tup[2][:,:,r,r] for r in 1:n]) #Eq 27
	obt_D, _ = eigen(obt_tilde)
	λT = sum(abs.(obt_D))
	mu_pos = (obt_D + abs.(obt_D))/2
	mu_neg = obt_D - mu_pos
	λTsqrt = (sum(mu_pos) - sum(mu_neg))

	FRAGS = run_optimization("g", tbt_mo_tup[2], CSA_tol, 1, 200, false, false, Float64[], Int64[], false, CSA_2_NAME, frag_flavour=CSA(), u_flavour=MF_real())
	α_CSA = length(FRAGS)

	CARTANS = zeros(α_CSA,n,n,n,n)
	for i in 1:α_CSA
		CARTANS[i,:,:,:,:] = fragment_to_normalized_cartan_tbt(FRAGS[i], frag_flavour=CSA())
	end
	global λV = 0.0
	global λVsqrt = 0.0
	tot_num = n_qubit
	for i in 1:α_CSA
		λ, Δ, num_u = cartan_to_qubit_l1_treatment(CARTANS[i,:,:,:,:], false)
		global λV += λ
		global λVsqrt += Δ
		tot_num += num_u 
	end
	println("Printout legend: 
		λ: 1-norm
		T: 1-body terms
		V: 2-body terms
		sqrt: square-root benchmarking unitarization
		p: symmetry-optimized
		tot_num: total number of unitaries
		α: total number of CSA fragments")
	@show λT, λV, λT+λV
	@show λTsqrt, λVsqrt, λTsqrt+λVsqrt
	@show tot_num, log2(tot_num)
	@show α_CSA+1, log2(α_CSA+1)

	println("Starting symmetry-after routine for CSA:")

	λs_arr = SharedArray(zeros(α_CSA,2))
	s_vec = zeros(3)
	tot_num = SharedArray(zeros(α_CSA))
	@sync @distributed for i in 1:α_CSA
		cartan_so = tbt_orb_to_so(CARTANS[i,:,:,:,:])
		#= PYTHON LINPROG ROUTINE
		cartan_triang_so = cartan_tbt_to_triang(cartan_so)
		sm, l1_orig, l1_red = qubit_sym_linprog_optimization(cartan_triang_so, n_qubit, true)
		tbt_cartan = cartan_so - shift_builder(sm, TB_Q_SYMS)
		# =#
		
		# = JULIA LINPROG ROUTINE
		sm = L1_linprog_optimizer_frag(cartan_so,SYM_MAT)
		tbt_cartan = cartan_so - shift_builder(sm, TB_Q_SYMS)
		s_vec += sm
		# =#

		λ, Δ, num_u = cartan_to_qubit_l1_treatment(tbt_cartan, true)
		λs_arr[i,:] .= [λ, Δ]
		tot_num[i] = num_u
	end
	tot_num = sum(tot_num) + n_qubit
	λVp = sum(λs_arr[:,1])
	λVpSqrt = sum(λs_arr[:,2])

	tot_shift = shift_builder(s_vec, TB_Q_SYMS)
	obt_so = obt_orb_to_so(obt_tilde) + 2*sum([tot_shift[:,:,r,r] for r in 1:n_qubit])
	ob_sol = ob_L1_optimization(obt_so)
	λTp = ob_sol.minimum
	obt_mod = obt_so - shift_builder(ob_sol.minimizer, OB_ARR)
	obt_D,_ = eigen(obt_mod)
	mu_pos = (obt_D + abs.(obt_D))/2
	mu_neg = obt_D - mu_pos
	λTpSqrt = 0.5 * (sum(mu_pos) - sum(mu_neg))

	@show λTp, λVp, λTp+λVp	
	@show λTpSqrt, λVpSqrt, λTpSqrt+λVpSqrt
	@show tot_num, log2(tot_num)
	
	#= Explicit shifting routine
	obt_shift = shift_builder(ob_sol.minimizer, OB_ARR)
	shift_op = of_simplify(h_ferm - tbt_to_ferm(tot_shift, true) - obt_to_ferm(obt_shift, true))
	shift_range = op_range(shift_op)
	println("Showing spectral range of total shifted Hamiltonian:")
	@show (shift_range[2] - shift_range[1])/2
	# =#
end

function DF_L1(tbt_mo_tup, h_ferm, ham_name, DF_tol=1e-6)
	# DO L1 CALCULATION ROUTINE USING DOUBLE FACTORIZATION DECOMPOSITION
	# INCLUDES BOTH WITHOUT AND WITH SYMMETRY OPTIMIZATION FOR INDIVIDUAL FRAGMENTS
	n_orbs = size(tbt_mo_tup[1])[1]
	n_qubit = 2*n_orbs
	OB_ARR, TB_ARR = casimirs_builder(n_qubit, one_body=true)
	TB_Q_SYMS = TB_ARR[3:end]
	SYM_MAT = τ_mat_builder(TB_Q_SYMS)


	println("Starting SVD routine for separated terms with Google's grouping technique:")
	CARTANS, TBTS = tbt_svd(tbt_mo_tup[2], tol=DF_tol, spin_orb=false, ret_op=false)
	α_SVD = size(CARTANS)[1]
	n = size(tbt_mo_tup[1])[1]
	obt_tilde = tbt_mo_tup[1] + 2*sum([tbt_mo_tup[2][:,:,r,r] for r in 1:n])
	obt_D, _ = eigen(obt_tilde)
	λT = sum(abs.(obt_D))
	mu_pos = (obt_D + abs.(obt_D))/2
	mu_neg = obt_D - mu_pos
	λTsqrt = (sum(mu_pos) - sum(mu_neg))

	global λV = 0.0
	global λVsqrt = 0.0
	tot_num = n_qubit
	for i in 1:α_SVD
		λ,Δ,num_u = cartan_to_qubit_l1_treatment(CARTANS[i,:,:,:,:], false)
		global λV += λ
		global λVsqrt += Δ
		tot_num += num_u
	end
	println("Printout legend: 
		λ: 1-norm
		T: 1-body terms
		V: 2-body terms
		sqrt: square-root benchmarking unitarization
		p: symmetry-optimized
		tot_num: total number of unitaries
		α: total number of SVD fragments")
	@show λT, λV, λT+λV
	@show λTsqrt, λVsqrt, λTsqrt+λVsqrt
	@show tot_num, log2(tot_num)
	@show α_SVD+1, log2(α_SVD+1)

	println("Starting symmetry-after routine for SVD:")

	λs_arr = SharedArray(zeros(α_SVD,2))
	s_vec = zeros(3)
	tot_num = SharedArray(zeros(α_SVD))
	for i in 1:α_SVD
		cartan_so = tbt_orb_to_so(CARTANS[i,:,:,:,:])

		#= PYTHON LINPROG OPTIMIZATION
		cartan_triang_so = cartan_tbt_to_triang(cartan_so)
		sm, l1_orig, l1_red = qubit_sym_linprog_optimization(cartan_triang_so, n_qubit, true)
		tbt_cartan = cartan_so - shift_builder(sm, TB_Q_SYMS)
		# =#

		# = JULIA LINPROG OPTIMIZATION
		sm = L1_linprog_optimizer_frag(cartan_so,SYM_MAT)
		tbt_cartan = cartan_so - shift_builder(sm, TB_Q_SYMS)
		# =#
		
		λ, Δ, num_u = cartan_to_qubit_l1_treatment(tbt_cartan, true)
		λs_arr[i,:] .= [λ, Δ]
		tot_num[i] = num_u
	end
	tot_num = sum(tot_num) + n_qubit
	λVp = sum(λs_arr[:,1])
	λVpSqrt = sum(λs_arr[:,2])

	tot_shift = shift_builder(s_vec, TB_Q_SYMS)
	obt_so = obt_orb_to_so(obt_tilde) + 2*sum([tot_shift[:,:,r,r] for r in 1:n_qubit])
	ob_sol = ob_L1_optimization(obt_so)
	λTp = ob_sol.minimum
	obt_mod = obt_so - shift_builder(ob_sol.minimizer, OB_ARR)
	obt_D,_ = eigen(obt_mod)
	mu_pos = (obt_D + abs.(obt_D))/2
	mu_neg = obt_D - mu_pos
	λTpSqrt = 0.5 * (sum(mu_pos) - sum(mu_neg))

	@show λTp, λVp, λTp+λVp
	@show λTpSqrt, λVpSqrt, λTpSqrt+λVpSqrt
	@show tot_num, log2(tot_num)

	#= Explicit shifting routine
	obt_shift = shift_builder(ob_sol.minimizer, OB_ARR)
	shift_op = of_simplify(h_ferm - tbt_to_ferm(tot_shift, true) - obt_to_ferm(obt_shift, true))
	shift_range = op_range(shift_op)
	println("Showing spectral range of total shifted Hamiltonian:")
	@show (shift_range[2] - shift_range[1])/2
	# =#
end

function COMBINED_L1(tbt_mo_tup, ham_name)
	# Calculates L1 norm by combining 1+2 body terms, less efficient...
	println("Starting SVD routine for combined 1+2e terms with Google's grouping technique:")
	tbt_so = tbt_to_so(tbt_mo_tup, false)
	CARTANS, TBTS = tbt_svd(tbt_so, tol=SVD_tol, spin_orb=true, ret_op=false)
	α_SVD = size(CARTANS)[1]
	n_so = 2n

	global obt_corr = zeros(n_so,n_so)
	for i in 1:α_SVD
		global obt_corr += 2*sum([TBTS[i,:,:,p,p] for p in 1:n_so])
	end
	obt_D, _ = eigen(obt_corr)
	λT = sum(abs.(obt_D))/2

	global λV = 0.0
	global λVsqrt = 0.0
	for i in 1:α_SVD
		λ, Δ = cartan_to_qubit_l1_treatment(CARTANS[i,:,:,:,:], true)
		global λV += λ
		global λVsqrt += Δ
	end
	@show λT, λV, λT+λV
	@show λVsqrt, λT+λVsqrt

	println("Starting CSA routine for combined 1+2e terms with Google's grouping technique:")
	tbt_so = tbt_to_so(tbt_mo_tup, false)
	CSA_12_NAME = "CSA_12_" * ham_name
	FRAGS = run_optimization("g", tbt_so, CSA_tol, 1, 200, false, false, Float64[], Int64[], true, CSA_12_NAME, frag_flavour=CSA(), u_flavour=MF_real())
	α_CSA = length(FRAGS)

	CARTANS = zeros(α_CSA,n_so,n_so,n_so,n_so)
	TBTS = zeros(α_CSA,n_so,n_so,n_so,n_so)
	for i in 1:α_CSA
		CARTANS[i,:,:,:,:] = fragment_to_normalized_cartan_tbt(FRAGS[i], frag_flavour=CSA())
		TBTS[i,:,:,:,:] = fragment_to_tbt(FRAGS[i], frag_flavour=CSA(), u_flavour=MF_real())
	end
	α_CSA = size(CARTANS)[1]

	global obt_corr = zeros(n_so,n_so)
	for i in 1:α_CSA
		global obt_corr += 2*sum([TBTS[i,:,:,p,p] for p in 1:n_so])
	end
	obt_D, _ = eigen(obt_corr)
	λT = sum(abs.(obt_D))/2

	global λV = 0.0
	global λVsqrt = 0.0
	for i in 1:α_CSA
		λ, Δ = cartan_to_qubit_l1_treatment(CARTANS[i,:,:,:,:], true)
		global λV += λ
		global λVsqrt += Δ
	end
	@show λT, λV, λT+λV
	@show λVsqrt, λT+λVsqrt
end

function SHIFTED_QUBIT_L1(tbt_mo_tup,transform=transformation)
	println("STARTING SYMMETRY OPTIMIZATIONS ROUTINE: FULL HAMILTONIAN SHIFTING")
	println("Shifting 1+2 term:")
	tbt_so = tbt_to_so(tbt_mo_tup, false)
	@time tbt_ham_opt, x_opt = symmetry_linprog_optimization(tbt_so, true)
	shift_op = tbt_to_ferm(tbt_ham_opt, true)
	shift_range = op_range(shift_op)
	println("Spectral range ΔE/2 after shift:")
	@show (shift_range[2] - shift_range[1])/2
	
	QUBIT_L1(shift_op,transform)
end


function FULL_TREATMENT(tbt_mo_tup, h_ferm, ham_name)
	QUBIT_L1(h_ferm,"jw")

	DF_tol = 1e-6
	CSA_tol = 1e-6

	println("\n\n\n")
	CSA_L1(tbt_mo_tup, h_ferm, ham_name, CSA_tol)
	
	println("\n\n\n")
	println("Starting Double-Factorization routine (reproduces results from Tensor HyperContraction paper, for benchmarking)")
	@time svd_optimized_df(tbt_mo_tup, tol=1e-6, tiny=1e-8)

	println("\n\n\n")
	DF_L1(tbt_mo_tup, h_ferm, ham_name, DF_tol)
	
	println("\n\n\n")
	SHIFTED_QUBIT_L1(tbt_mo_tup,"jw")

	return 0
end

function REDUCED_TREATMENT(tbt_mo_tup, h_ferm, ham_name)
	println("Starting qubit treatment...")
	# =
	println("Performing fermion to qubit mapping:")
	@time H_full_q = qubit_transform(h_ferm, "jw")
	println("\n\n\n Qubit treatment of Hamiltonian:")
	qubit_treatment(H_full_q)

	
	SVD_tol = 1e-6
	CSA_tol = 1e-6

	n_qubit = 2*size(tbt_mo_tup[1])[1]
	
	println("\n\n Starting CSA routine for separated 1 and 2-body terms")
	CSA_2_NAME = "CSA_2_" * ham_name 
	n = size(tbt_mo_tup[1])[1]
	obt_tilde = tbt_mo_tup[1] + 2*sum([tbt_mo_tup[2][:,:,r,r] for r in 1:n])
	obt_D, _ = eigen(obt_tilde)
	λT = sum(abs.(obt_D))
	mu_pos = (obt_D + abs.(obt_D))/2
	mu_neg = obt_D - mu_pos
	λTsqrt = (sum(mu_pos) - sum(mu_neg))

	FRAGS = run_optimization("g", tbt_mo_tup[2], CSA_tol, 1, 200, false, false, Float64[], Int64[], false, CSA_2_NAME, frag_flavour=CSA(), u_flavour=MF_real())
	α_CSA = length(FRAGS)

	CARTANS = zeros(α_CSA,n,n,n,n)
	for i in 1:α_CSA
		CARTANS[i,:,:,:,:] = fragment_to_normalized_cartan_tbt(FRAGS[i], frag_flavour=CSA())
	end
	global λV = 0.0
	global λVsqrt = 0.0
	tot_num = n_qubit
	for i in 1:α_CSA
		λ, Δ, num_u = cartan_to_qubit_l1_treatment(CARTANS[i,:,:,:,:], false)
		global λV += λ
		global λVsqrt += Δ
		tot_num += num_u 
	end
	@show λT, λV, λT+λV
	@show λTsqrt, λVsqrt, λTsqrt+λVsqrt
	@show tot_num, log2(tot_num)
	@show α_CSA+1, log2(α_CSA+1)

	
	println("\n\n\n Starting df routine...")
	@time svd_optimized_df(tbt_mo_tup, tol=1e-6, tiny=1e-8)

	println("\n\n\n			Starting SVD routine for separated terms with Google's grouping technique:")
	CARTANS, TBTS = tbt_svd(tbt_mo_tup[2], tol=SVD_tol, spin_orb=false, ret_op=false)
	α_SVD = size(CARTANS)[1]
	n = size(tbt_mo_tup[1])[1]
	obt_tilde = tbt_mo_tup[1] + 2*sum([tbt_mo_tup[2][:,:,r,r] for r in 1:n])
	obt_D, _ = eigen(obt_tilde)
	λT = sum(abs.(obt_D))
	mu_pos = (obt_D + abs.(obt_D))/2
	mu_neg = obt_D - mu_pos
	λTsqrt = (sum(mu_pos) - sum(mu_neg))

	global λV = 0.0
	global λVsqrt = 0.0
	tot_num = n_qubit
	for i in 1:α_SVD
		λ,Δ,num_u = cartan_to_qubit_l1_treatment(CARTANS[i,:,:,:,:], false)
		global λV += λ
		global λVsqrt += Δ
		tot_num += num_u
	end
	@show λT, λV, λT+λV
	@show λTsqrt, λVsqrt, λTsqrt+λVsqrt
	@show tot_num, log2(tot_num)
	@show α_SVD+1, log2(α_SVD+1)

	return 0
end

function op_treatment(OP) #FermionOperator
	r = op_range(OP)
	@show (r[2] - r[1])/2
	println("Performing fermion to qubit mapping:")
	@time H_full_q = qubit_transform(OP, "jw")
	println("\n\n\n Qubit treatment of Hamiltonian:")
	qubit_treatment(H_full_q)
end

function MAJORANA_SHIFTED_TREATMENT(tbt_mo_tup, reps=10)
	n = size(tbt_mo_tup[1])[1]
	S_arr  = casimirs_builder(2n, S2=false)
	
	println("Finding shift...")
	tbt_so = tbt_to_so(tbt_mo_tup, false)
	@time tbt_ham_opt, x_opt = symmetry_linprog_optimization(tbt_so, true)
	@show x_opt

	shift_op = tbt_to_ferm(tbt_ham_opt, true)
	op_treatment(shift_op)
	

	println("\n\n\n\n\n##################################\n
		Starting Majorana treatment for full 5-dim shift:")
	shift_vec = [x_opt[3], x_opt[5]]
	@time tbt_rot_tup = post_shift_majorana_optimization(tbt_mo_tup, shift_vec, reps)
	shift_tbt = shift_builder(x_opt, S_arr)
	shift_op = tbt_to_ferm(tbt_rot_tup, false) - tbt_to_ferm(shift_tbt, true)
	println("\n####\n
		Starting full shift treatment...")
	op_treatment(shift_op)

	shift_tbt2 = sum(x_opt[3:5] .* S_arr[3:5])
	shift_op = tbt_to_ferm(tbt_rot_tup, false) - tbt_to_ferm(shift_tbt, true)
	println("\n####\n
		Starting 2-body shift treatment...")
	op_treatment(shift_op)

	println("\n\n\n\n\n##################################\n
		Starting Majorana treatment for Cartan 2-dim shift:")
	tbt_shift, s_vec = cartan_tbt_purification(tbt_mo_tup[2], false)
	@show s_vec
	shift_vec = [s_vec[1], s_vec[3]]
	@time tbt_rot_tup = post_shift_majorana_optimization(tbt_mo_tup, shift_vec, reps)
	shift_tbt = sum(s_vec .* S_arr[3:end])
	shift_op = tbt_to_ferm(tbt_rot_tup, false) - tbt_to_ferm(shift_tbt, true)
	op_treatment(shift_op)
	ham_shift = tbt_to_ferm(tbt_mo_tup, false) - tbt_to_ferm(shift_tbt, true)
	op_treatment(ham_shift)

	println("\n\n\n\n\n##################################\n
		Starting Majorana treatment for full l2 norm shift:")
	s_l2 = cholesky_symmetry_l2_optimization(tbt_mo_tup[2])
	@show s_l2
	shift_vec = [s_l2, s_l2]
	@time tbt_rot_tup = post_shift_majorana_optimization(tbt_mo_tup, shift_vec, reps)
	shift_tbt = s_l2 * sum(S_arr[3:end])
	shift_op = tbt_to_ferm(tbt_rot_tup, false) - tbt_to_ferm(shift_tbt, true)
	op_treatment(shift_op)

	println("\n\n\n\n\n##################################\n
		Starting Majorana treatment for full l1 norm shift:")
	s_l1 = cholesky_symmetry_l1_optimization(tbt_mo_tup[2])
	@show s_l1
	shift_vec = [s_l1, s_l1]
	@time tbt_rot_tup = post_shift_majorana_optimization(tbt_mo_tup, shift_vec, reps)
	shift_tbt = s_l1 * sum(S_arr[3:end])
	shift_op = tbt_to_ferm(tbt_rot_tup, false) - tbt_to_ferm(shift_tbt, true)
	op_treatment(shift_op)
end

