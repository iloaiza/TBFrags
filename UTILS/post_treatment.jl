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
	#λs[:,1] = [λij]
	#λs[:,2] = [ni num]
	#λs[:,3] = [nj num]
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

function py_sparse_import(py_sparse_mat; imag_tol=1e-16)
	#transform python sparse matrix into julia sparse matrix
	row, col, vals = scipy.sparse.find(py_sparse_mat)
	py_shape = py_sparse_mat.get_shape()
	n = py_shape[1]
	
	if sum(imag.(vals)) < imag_tol
		vals = real.(vals)
	end

	sparse_mat = sparse(row .+1, col .+1, vals, n, n)

	#@show sum(abs2.(sparse_mat - py_sparse_mat.todense()))

	return sparse_mat
end

function qubit_op_range(op_qubit, n_qubit=of.count_qubits(op_qubit); imag_tol=1e-16, ncv=minimum([50,2^n_qubit]))
	#Calculates maximum and minimum eigenvalues for qubit operator
	op_py_sparse_mat = of.qubit_operator_sparse(op_qubit)
	sparse_op = py_sparse_import(op_py_sparse_mat, imag_tol=imag_tol)

	# =
	if n_qubit >= 2
		E_max,_ = eigs(sparse_op, nev=1, which=:LR, maxiter = 500, tol=1e-3, ncv=ncv)
		E_min,_ = eigs(sparse_op, nev=1, which=:SR, maxiter = 500, tol=1e-3, ncv=ncv)
	else
		E,_ = eigen(collect(sparse_op))
		E = real.(E)
		E_max = maximum(E)
		E_min = minimum(E)
	end
	E_range = real.([E_min[1], E_max[1]])
	# =#

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

function L1_frags_treatment(TBTS, CARTAN_TBTS, spin_orb, α_tot = size(TBTS)[1], n=size(TBTS)[2])
	#Caclulate different L1 norms for group of fragments
	CSA_L1 = SharedArray(zeros(α_tot)) #holds (sum _ij |λij|) L1 norm for CSA polynomial
	E_RANGES = SharedArray(zeros(α_tot,2)) #holds eigenspectrum boundaries for each operator
	L2 = SharedArray(zeros(α_tot)) #holds L2 fermionic norm of each tbt
	
	@sync @distributed for i in 1:α_tot
		#println("L1 treatment of fragment $i")
		#t00 = time()
		#build qubit operator for final sanity check
		
		CSA_L1[i] = cartan_tbt_l1_cost(CARTAN_TBTS[i,:,:,:,:], spin_orb)
		L2[i] = tbt_cost(TBTS[i,:,:,:,:], 0)
		
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

	return CSA_L1, E_RANGES, L2
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

function H_POST(tbt, h_ferm, x0, K0, spin_orb; frag_flavour=META.ff, Q_TREAT=true, S2=true)
	#builds fragments from x0 and K0, and compares with full operator h_ferm and its associated tbt
	tbt_so = tbt_to_so(tbt, spin_orb)
	println("Obtaining symmetry-shifted Hamiltonian")
	@time tbt_ham_opt, x_opt = symmetry_cuadratic_optimization(tbt_so, true, S2=S2)
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
		shift = shift_builder(pur_coeffs_svd, S_arr, S2=S2)
		PUR_SVD_TBTS[i,:,:,:,:] = SVD_TBTS[i,:,:,:,:] - shift
	end

	x = zeros(5)
	for i in 1:5
		x = PUR_SVD_COEFFS[:,i]
	end
	shift = shift_builder(x, S_arr, S2=S2)
	H_SYM_FERM_SVD = SVD_OP - tbt_to_ferm(shift, true)
	# =#

	S_arr = casimirs_builder(n_qubit, S2=true)

	println("Starting L1 treatment for SVD fragments")
	SVD_L1, SVD_E_RANGES,_ = L1_frags_treatment(SVD_TBTS, SVD_CARTAN_TBTS, true)
	#SVD_PUR_L1, SVD_PUR_RANGES, _ = L1_frags_treatment(SVD_TBTS, PUR_SVD_CARTANS, true)
	#= CSA subsection
	CSA_L1, PUR_L1, E_RANGES, PUR_RANGES, CSA_OP = L1_frags_treatment(TBTS, CARTAN_TBTS, PUR_CARTANS, spin_orb)
	global H_SYM_FERM = of.FermionOperator.zero()
	for i in 1:α_tot
		x = PUR_COEFFS[i]
		shift = shift_builder(x, S_arr, S2=S2)
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
	full_shift = shift_builder(x_opt, S_arr, S2=S2)
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


function H_TREATMENT(tbt, h_ferm, spin_orb; Q_TREAT=true, S2=true)
	tbt_so = tbt_to_so(tbt, spin_orb)
	println("Obtaining symmetry-shifted Hamiltonian")
	@time tbt_ham_opt, x_opt = symmetry_cuadratic_optimization(tbt_so, true, S2=S2)
	@show x_opt
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
	@time SVD_L1, SVD_E_RANGES, SVD_L2 = L1_frags_treatment(SVD_TBTS, SVD_CARTAN_TBTS, true)
	println("Starting L1 treatment for SVD fragments of purified Hamiltonian")
	@time PUR_SVD_L1, PUR_SVD_E_RANGES, PUR_SVD_L2 = L1_frags_treatment(PUR_SVD_TBTS, PUR_SVD_CARTAN_TBTS, true)

	println("Starting purification and L1 treatment for SVD fragments:")
	SVD_THEN_PUR_CARTAN_TBTS = zeros(α_SVD, n_qubit, n_qubit, n_qubit, n_qubit)
	SVD_THEN_PUR_TBTS = zeros(α_SVD, n_qubit, n_qubit, n_qubit, n_qubit)
	X_OPTS = zeros(α_SVD, length(x_opt))
	TOT_SHIFT = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	for i in 1:α_SVD
		SVD_THEN_PUR_TBTS[i,:,:,:,:], X_OPTS[i,:] = symmetry_cuadratic_optimization(SVD_TBTS[i,:,:,:,:], true, S2=S2)
		shift = shift_builder(X_OPTS[i,:], S_arr, S2=S2)
		SVD_THEN_PUR_CARTAN_TBTS[i,:,:,:,:] = SVD_CARTAN_TBTS[i,:,:,:,:] - shift
		TOT_SHIFT += shift
	end
	if S2 == true
		println("Warning, Cartan norm will give wrong results for SVD-then-shifted since S² is not a Cartan form")
	end
	SVD_THEN_PUR_L1, SVD_THEN_PUR_E_RANGES, SVD_THEN_PUR_L2 = L1_frags_treatment(SVD_THEN_PUR_TBTS, SVD_THEN_PUR_CARTAN_TBTS, true)

	println("Starting purification and L1 treatment for SVD fragments of symmetry-optimized H:")
	PUR2_CARTAN_TBTS = zeros(PUR_α_SVD, n_qubit, n_qubit, n_qubit, n_qubit)
	PUR2_TBTS = zeros(PUR_α_SVD, n_qubit, n_qubit, n_qubit, n_qubit)
	TOT_SHIFT_2 = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	X_OPTS_2 = zeros(PUR_α_SVD, length(x_opt))
	for i in 1:PUR_α_SVD
		PUR2_TBTS[i,:,:,:,:], X_OPTS_2[i,:] = symmetry_cuadratic_optimization(PUR_SVD_TBTS[i,:,:,:,:], true, S2=S2)
		shift = shift_builder(X_OPTS_2[i,:], S_arr, S2=S2)
		PUR2_CARTAN_TBTS[i,:,:,:,:] = PUR_SVD_CARTAN_TBTS[i,:,:,:,:] - shift
		TOT_SHIFT_2 += shift
	end
	if S2 == true
		println("Warning, Cartan norm will give wrong results for SVD-then-shifted since S² is not a Cartan form")
	end
	PUR2_L1, PUR2_E_RANGES, PUR2_L2 = L1_frags_treatment(PUR2_TBTS, PUR2_CARTAN_TBTS, true)

	
	println("Calculating range of full hamiltonian:")
	@time Etot_r = op_range(h_ferm)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2

	println("Showing range of symmetry-optimized hamiltonian")
	H_sym_opt = tbt_to_ferm(tbt_ham_opt, true)
	@time Etot_r = op_range(H_sym_opt)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	full_shift = shift_builder(x_opt, S_arr, S2=S2)

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

function QUBIT_TREATMENT(tbt, h_ferm, spin_orb; S2=true)
	#builds fragments from x0 and K0, and compares with full operator h_ferm and its associated tbt
	tbt_so = tbt_to_so(tbt, spin_orb)
	println("Obtaining symmetry-shifted Hamiltonian")
	@time tbt_ham_opt, x_opt = symmetry_cuadratic_optimization(tbt_so, true, S2=S2)
	@show x_opt
	#@time tbt_ham_opt_SD_S2, x_opt_SD_S2, u_params_S2 = orbital_mean_field_symmetry_reduction(tbt,spin_orb, u_flavour=MF_real(), S2=true, cartan=false)
	#@show x_opt_SD

	#@time tbt_ham_opt_SD, x_opt_SD, u_params = orbital_mean_field_symmetry_reduction(tbt, spin_orb, u_flavour=MF_real(), S2=false, cartan=false)
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
	#full_shift = shift_builder(x_opt, S_arr, S2=S2)

	#=
	println("Showing range of unitary + symmetry-optimized hamiltonian")
	H_sym_opt_SD = tbt_to_ferm(tbt_ham_opt_SD, true)
	@time Etot_r = op_range(H_sym_opt_SD)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	#full_shift = shift_builder(x_opt, S_arr, S2=S2)
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
