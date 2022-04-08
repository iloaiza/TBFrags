function op_range(op, n_qubit; transformation=transformation, imag_tol=1e-16)
	#Calculates maximum and minimum eigenvalues for fermionic operator
	op_qubit = qubit_transform(op)
	op_py_sparse_mat = of.qubit_operator_sparse(op_qubit)
	sparse_op = spzeros(2^n_qubit, 2^n_qubit)
	rows, cols = op_py_sparse_mat.nonzero()
	IMAG_FLAG = false
	for el_num in 1:length(rows)
		r_num = rows[el_num]
		c_num = cols[el_num]
		if imag(op_py_sparse_mat[r_num,c_num]) < imag_tol
			sparse_op[r_num+1, c_num+1] = real(op_py_sparse_mat[r_num,c_num])
		else
			IMAG_FLAG = true
			break
		end
	end

	if IMAG_FLAG == true
		sparse_op = spzeros(Complex{Float64}, 2^n_qubit, 2^n_qubit)
		for el_num in 1:length(rows)
			r_num = rows[el_num]
			c_num = cols[el_num]
			sparse_op[r_num+1, c_num+1] = op_py_sparse_mat[r_num,c_num]
		end
	end

	# =
	E_max,_ = eigs(sparse_op, nev=1, which=:LR, maxiter = 500000, tol=1e-6)
	E_min,_ = eigs(sparse_op, nev=1, which=:SR, maxiter = 500000, tol=1e-6)
	E_range = real.([E_min[1], E_max[1]])
	# =#

	#=
	E, _ = eigen(collect(sparse_op))
	E = real.(E)
	E_range = [minimum(E), maximum(E)]
	# =#
	
	
	return E_range
end

mutable struct Atomic{T}; @atomic x::T; end

function L1_frags_treatment(TBTS, CARTAN_TBTS, PUR_CARTANS, spin_orb, α_tot = size(TBTS)[1])
	#Caclulate different L1 norms for group of fragments
	TOT_OP = Atomic(of.FermionOperator.zero()) #holds final operator which is sum of all fragments, used for sanity check
	CSA_L1 = SharedArray(zeros(α_tot)) #holds (sum _ij |λij|) L1 norm for CSA polynomial
	PUR_L1 = SharedArray(zeros(α_tot)) #same as CSA_L1 but for symmetry purified polynomials
	E_RANGES = SharedArray(zeros(α_tot,2)) #holds eigenspectrum boundaries for each operator
	PUR_RANGES = SharedArray(zeros(α_tot,2)) #same as E_RANGES but for purified polynomials
	
	@sync @distributed for i in 1:α_tot
		println("L1 treatment of fragment $i")
		#build qubit operator for final sanity check
		#global TOT_OP += tbt_to_ferm(TBTS[i], spin_orb)
		@atomic TOT_OP.x += tbt_to_ferm(TBTS[i,:,:,:,:], spin_orb)
		
		CSA_L1[i] = cartan_tbt_l1_cost(CARTAN_TBTS[i,:,:,:,:], spin_orb)
		PUR_L1[i] = cartan_tbt_l1_cost(PUR_CARTANS[i,:,:,:,:], true)

		# SQRT_L1 subroutine
		op_CSA = tbt_to_ferm(CARTAN_TBTS[i,:,:,:,:], spin_orb)
		E_RANGES[i,:] = real.(op_range(op_CSA, n_qubit))
		
		op_CSA_red = tbt_to_ferm(PUR_CARTANS[i,:,:,:,:], true)
		PUR_RANGES[i,:] = real.(op_range(op_CSA_red, n_qubit))
		
		#= linear programing reflection optimization
		obt_CSA_mo, tbt_CSA_mo = CARTAN_TBTS[i]
		tbt_CSA_mo = cartan_tbt_to_triang(tbt_CSA_mo)

		@time ref_sol = car2lcu.OBTTBT_to_L1opt_LCU(2obt_CSA_mo, 4tbt_CSA_mo, n, solmtd="l1ip", pout=false)
		CSA_L1_MO[i] = ref_sol["csa_l1"]
		CR_L1_MO[i] = ref_sol["lcu_l1"]	
		CR_FRAGS_MO[i] = ref_sol["poldim"]

		# =#
		# SPIN-ORBIT REFLECTIONS ROUTINE
		#=
		if spin_orb == true
			obt_CSA_so, tbt_CSA_so = CARTAN_TBTS[i]
			tbt_CSA_so += obt_to_tbt(obt_CSA_so)
		else
			tbt_CSA_so = tbt_orb_to_so(CARTAN_TBTS[i][2])
			obt_CSA_so = obt_orb_to_so(CARTAN_TBTS[i][1])
			tbt_CSA_so += cartan_obt_to_tbt(obt_CSA_so)

			# DEBUG: CHECK BUILT TBT GIVES BACK SAME FERMIONIC OPERATOR
			#op_CSA = tbt_to_ferm(CARTAN_TBTS[i], false)		
			#@show of_simplify(op_CSA - tbt_to_ferm(tbt_CSA_so, true))
		end

		tbt_triang = cartan_tbt_to_triang(tbt_CSA_so)
		#@show of_simplify(tbt_to_ferm(tbt_CSA_so, true) - tbt_to_ferm(tbt_triang, true))
		println("Starting full two-body, spin-orb optimization")
		@time ref_sol = car2lcu.TBT_to_L1opt_LCU(tbt_triang, n_qubit, solmtd="l1ip", pout=true)
		CSA_L1[i] = ref_sol["csa_l1"]
		CR_L1[i] = ref_sol["lcu_l1"]
		# =#
	end


	return CSA_L1, PUR_L1, E_RANGES, PUR_RANGES, @atomic TOT_OP.x
end

function qubit_treatment(H_q)
	#does quantum treatment of H_q qubit Hamiltonian
	println("Starting AC-RLF decomposition")
	@time op_list_AC, L1_AC, Pauli_cost_AC, Pauli_num = anti_commuting_decomposition(H_q)
	@show L1_AC, Pauli_cost_AC, length(op_list_AC)

	println("Starting AC-SI decomposition")
	@time op_list, L1_sorted, Pauli_cost = ac_sorted_inversion(H_q)
	@show L1_sorted, length(op_list)
	println("Pauli=$(Pauli_cost)($(ceil(log2(Pauli_num))))")
	println("AC-RLF L1=$(L1_AC)($(ceil(log2(length(op_list_AC)))))")
	println("AC-SI L1=$(L1_sorted)($(ceil(log2(length(op_list)))))")
end

function H_POST(tbt, h_ferm, x0, K0, spin_orb; frag_flavour=META.ff, Q_TREAT=true)
	#builds fragments from x0 and K0, and compares with full operator h_ferm and its associated tbt
	tbt_so = tbt_to_so(tbt, spin_orb)
	SVD_CARTAN_TBTS, SVD_TBTS = tbt_svd(tbt_so, tol=1e-6, spin_orb=true, return_CSA=true)
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
	TBTS = []
	CARTAN_TBTS = []
	PUR_CARTANS = [] #purified Cartan tbts as two-body tensors in spin-orbitals
	PUR_COEFFS = []

	fcl = frag_coeff_length(n, frag_flavour)
	global x_ini = 1
	println("Starting fragment building and purification for CSA...")
	for i in 1:α_tot
		x_curr = x0[x_ini:x_ini+x_size-1]
		frag = fragment(x_curr[fcl+1:end], x_curr[1:fcl], K0[i], n_qubit, spin_orb)
		push!(FRAGS, frag)
		push!(TBTS, fragment_to_tbt(frag))
		push!(CARTAN_TBTS, fragment_to_normalized_cartan_tbt(frag))
		pur_tbt, pur_coeffs = cartan_tbt_purification(CARTAN_TBTS[i], spin_orb)
		push!(PUR_CARTANS, pur_tbt)
		push!(PUR_COEFFS, pur_coeffs)
		global x_ini += x_size
	end
	# =#

	PUR_SVD_CARTANS = SharedArray(zeros(Complex{Float64}, α_SVD, n_qubit, n_qubit, n_qubit, n_qubit))
	PUR_SVD_COEFFS = SharedArray(zeros(Complex{Float64}, α_SVD, 5))
	@sync @distributed for i in 1:α_SVD
		println("Purifying Cartan fragment $i")
		@time pur_tbt_svd, pur_coeffs_svd = cartan_tbt_purification(SVD_CARTAN_TBTS[i,:,:,:,:], true)
		PUR_SVD_CARTANS[i,:,:,:,:] = pur_tbt_svd
		PUR_SVD_COEFFS[i,:] = pur_coeffs_svd
	end

	_, _, Nα_tbt, Nβ_tbt, Nα2_tbt, NαNβ_tbt, Nβ2_tbt = casimirs_builder(n_qubit)

	println("Starting L1 treatment for SVD fragments")
	SVD_L1, SVD_PUR_L1, SVD_E_RANGES, SVD_PUR_RANGES, SVD_OP = L1_frags_treatment(SVD_TBTS, SVD_CARTAN_TBTS, PUR_SVD_CARTANS, true)
	
	#= CSA subsection
	CSA_L1, PUR_L1, E_RANGES, PUR_RANGES, CSA_OP = L1_frags_treatment(TBTS, CARTAN_TBTS, PUR_CARTANS, spin_orb)
	global H_SYM_FERM = of.FermionOperator.zero()
	H_SYM_FERM += h_ferm
	for i in 1:α_tot
		x = PUR_COEFFS[i]
		shift = x[1]*Nα_tbt + x[2]*Nβ_tbt + x[3]*Nα2_tbt + x[4]*NαNβ_tbt + x[5]*Nβ2_tbt
		global H_SYM_FERM -= tbt_to_ferm(shift, true)
	end_qubit
	H_SYM_FERM = of_simplify(H_SYM_FERM)
	# =#

	global H_SYM_FERM_SVD = of.FermionOperator.zero()
	H_SYM_FERM_SVD += h_ferm
	for i in 1:α_SVD
		x = PUR_SVD_COEFFS[i,:]
		shift = x[1]*Nα_tbt + x[2]*Nβ_tbt + x[3]*Nα2_tbt + x[4]*NαNβ_tbt + x[5]*Nβ2_tbt
		global H_SYM_FERM_SVD -= tbt_to_ferm(shift, true)
	end
	H_SYM_FERM_SVD = of_simplify(H_SYM_FERM_SVD)

	println("Calculating range of full hamiltonian:")
	Etot_r = op_range(h_ferm, n_qubit)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2

	#= CSA
	println("Calculating range of symmetry shifted hamiltonian (CSA):")
	Etot_r = op_range(H_SYM_FERM, n_qubit)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	# =#

	println("Calculating range of symmetry shifted hamiltonian (SVD):")
	Etot_r = op_range(H_SYM_FERM_SVD, n_qubit)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2

	println("Optimizing and showing range of symmetry-optimized hamiltonian")
	tbt_ham_opt, x = cartan_tbt_purification(tbt_so, true)
	H_sym_opt = tbt_to_ferm(tbt_ham_opt, true)
	Etot_r = op_range(H_sym_opt, n_qubit)
	@show Etot_r
	@show (Etot_r[2] - Etot_r[1])/2
	shift = x[1]*Nα_tbt + x[2]*Nβ_tbt + x[3]*Nα2_tbt + x[4]*NαNβ_tbt + x[5]*Nβ2_tbt
	@show of_simplify(h_ferm - H_sym_opt - tbt_to_ferm(shift, true))

	println("CSA L1 bounds (NR):")
	#@show sum(CSA_L1)/2
	@show sum(SVD_L1)/2
	println("Shifted minimal norm (SR):")
	#ΔE_CSA = [(E_RANGES[i,2] - E_RANGES[i,1])/2 for i in 1:size(E_RANGES)[1]]
	#@show sum(ΔE_CSA)
	ΔE_SVD = [(SVD_E_RANGES[i,2] - SVD_E_RANGES[i,1])/2 for i in 1:size(SVD_E_RANGES)[1]]
	@show sum(ΔE_SVD)
	println("Purified L1 bounds (S-NR):")
	#@show sum(PUR_L1)/2
	@show sum(SVD_PUR_L1)/2
	println("Purified minimal norm (SS):")
	#ΔE_CSA_PUR = [(PUR_RANGES[i,2] - PUR_RANGES[i,1])/2 for i in 1:size(PUR_RANGES)[1]]
	#@show sum(ΔE_CSA_PUR)
	ΔE_SVD_PUR = [(SVD_PUR_RANGES[i,2] - SVD_PUR_RANGES[i,1])/2 for i in 1:size(SVD_PUR_RANGES)[1]]
	@show sum(ΔE_SVD_PUR)

	if Q_TREAT == true
		println("Finished fermionic routine, starting qubit methods and final numbers...")
		H_full_bk = qubit_transform(h_ferm)
		#H_sym_bk = qubit_transform(H_SYM_FERM)
		H_svd_bk = qubit_transform(H_SYM_FERM_SVD)
		H_opt_bk = qubit_transform(H_sym_opt)
		#H_tapered = tap.taper_H_qubit(H_full_bk)
		#H_tapered_sym = tap.taper_H_qubit(H_sym_bk)

		println("Full Hamiltonian:")
		qubit_treatment(H_full_bk)

		#println("CSA shifted Hamiltonian:")
		#qubit_treatment(H_sym_bk)

		println("SVD shifted Hamiltonian:")
		qubit_treatment(H_svd_bk)

		println("Optimal shifted Hamiltonian:")
		qubit_treatment(H_opt_bk)

		#exit()
		#sanity check, final operator recovers full Hamiltonian
		H_diff = h_ferm - SVD_OP
		H_qubit_diff = qubit_transform(H_diff)
		
		println("Difference from CSA fragments:")
		@show qubit_operator_trimmer(H_qubit_diff, 1e-5)

		println("Difference from SVD fragments:")
		H_diff_SVD = h_ferm - SVD_OP
		H_qubit_diff_SVD = qubit_transform(H_diff_SVD)
		@show qubit_operator_trimmer(H_qubit_diff_SVD, 1e-5)
	end
end
