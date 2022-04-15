#reduces Cartan polynomials using Casimir operators
function casimirs_builder(n_qubit; debug=false, S2=false)
	n_orbs = Int(n_qubit/2)
	N_obt = collect(Diagonal(ones(n_qubit)))
	Sz_obt = 0.5*copy(N_obt)
	for i in 2:2:n_qubit
		Sz_obt[i,i] *= -1
	end
	Nα_obt = 0.5*(N_obt + 2*Sz_obt)
	Nβ_obt = N_obt - Nα_obt

	N_tbt = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	for i in 1:n_qubit
		N_tbt[i,i,i,i] = 1
	end

	Sz_tbt = 0.5*copy(N_tbt)
	for i in 2:2:n_qubit
		Sz_tbt[i,i,i,i] *= -1
	end

	Nα_tbt = 0.5*(N_tbt + 2*Sz_tbt)
	Nβ_tbt = N_tbt - Nα_tbt	

	N2_tbt = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	for i in 1:n_qubit
		for j in 1:n_qubit
			N2_tbt[i,i,j,j] = 1
		end
	end

	NαNβ_tbt = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	for i in 1:n_orbs
		for j in 1:n_orbs
			NαNβ_tbt[2i-1,2i-1,2j,2j] = 0.5
			NαNβ_tbt[2j,2j,2i-1,2i-1] = 0.5
		end
	end

	Nα2_tbt = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	for i in 1:n_orbs
		for j in 1:n_orbs
			Nα2_tbt[2i-1,2i-1,2j-1,2j-1] = 1
		end
	end

	Nβ2_tbt = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	for i in 1:n_orbs
		for j in 1:n_orbs
			Nβ2_tbt[2i,2i,2j,2j] = 1
		end
	end


	if debug == true
		N_op = of.number_operator(n_qubit)
		N2 = N_op * N_op
		Sz = of.sz_operator(n_orbs)
		N_α = 0.5*of_simplify(N_op + 2*Sz)
		N_β = of_simplify(N_op - N_α)
		@show of_simplify(N_op - obt_to_ferm(N_obt, true))
		@show of_simplify(N_α - obt_to_ferm(Nα_obt, true))
		@show of_simplify(N_β - obt_to_ferm(Nβ_obt, true))
		@show of_simplify(Sz - obt_to_ferm(Sz_obt, true))
		@show of_simplify(N_op - tbt_to_ferm(N_tbt, true))
		@show of_simplify(N_α - tbt_to_ferm(Nα_tbt, true))
		@show of_simplify(N_β - tbt_to_ferm(Nβ_tbt, true))
		@show of_simplify(Sz - tbt_to_ferm(Sz_tbt, true))
		@show of_simplify(N2 - tbt_to_ferm(N2_tbt, true))
		@show of_simplify(N_α*N_β - tbt_to_ferm(NαNβ_tbt, true))
		@show of_simplify(N_α*N_α - tbt_to_ferm(Nα2_tbt, true))
		@show of_simplify(N_β*N_β - tbt_to_ferm(Nβ2_tbt, true))
	end
	

	if S2 == false
		return [Nα_tbt, Nβ_tbt, Nα2_tbt, NαNβ_tbt, Nβ2_tbt]
	else
		S2_tbt = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
		for i in 1:n_orbs
			ka = 2i-1
			kb = 2i
			S2_tbt[ka,ka,ka,ka] += 1
			S2_tbt[kb,kb,kb,kb] += 1
			S2_tbt[ka,ka,kb,kb] += -1
			S2_tbt[kb,kb,ka,ka] += -1
			S2_tbt[ka,kb,kb,ka] += 2
			S2_tbt[kb,ka,ka,kb] += 2
		end

		for i in 1:n_orbs
			for j in 1:n_orbs
				if i != j
					ka = 2i-1
					kb = 2i
					la = 2j-1
					lb = 2j
					S2_tbt[ka,kb,lb,la] += 1
					S2_tbt[lb,la,ka,kb] += 1
					S2_tbt[kb,ka,la,lb] += 1
					S2_tbt[la,lb,kb,ka] += 1
					S2_tbt[ka,ka,la,la] += 0.5
					S2_tbt[la,la,ka,ka] += 0.5
					S2_tbt[kb,kb,lb,lb] += 0.5
					S2_tbt[lb,lb,kb,kb] += 0.5
					S2_tbt[ka,ka,lb,lb] += -0.5
					S2_tbt[lb,lb,ka,ka] += -0.5
					S2_tbt[kb,kb,la,la] += -0.5
					S2_tbt[la,la,kb,kb] += -0.5
				end
			end
		end

		S2_tbt /= 4

		if debug == true
			S2_op = of.s_squared_operator(n_orbs)
			@show of_simplify(S2_op - tbt_to_ferm(S2_tbt, true))
		end

		return [Nα_tbt, Nβ_tbt, Nα2_tbt, NαNβ_tbt, Nβ2_tbt, S2_tbt]
	end
end

function shift_builder(x, S_arr; S2=false)
	# S2 = false implies S_arr = [Nα_tbt, Nβ_tbt, Nα2_tbt, NαNβ_tbt, Nβ2_tbt]
	# otherwise S_arr = [Nα_tbt, Nβ_tbt, Nα2_tbt, NαNβ_tbt, Nβ2_tbt, S2_tbt]
	# builds shift corresponding to x value
	
	shift = x[1] * S_arr[1]

	for i in 2:length(S_arr)
		shift += x[i] * S_arr[i]
	end

	return shift
end

function cartan_tbt_purification(tbt :: Array, spin_orb=true)
	# input: cartan tbt operator (and whether it is in spin-orbitals or orbitals)
	# output: cartan tbt operator in spin-orbitals with shifted symmetries, and shift constants
	if spin_orb == false
		tbt_so = tbt_orb_to_so(tbt)
	else
		tbt_so = tbt
	end

	n_qubit = size(tbt_so)[1]

	S_arr = casimirs_builder(n_qubit, S2=false)

	function cost(x) 
		#shift = x[1]*Nα_tbt + x[2]*Nβ_tbt + x[3]*Nα2_tbt + x[4]*NαNβ_tbt + x[5]*Nβ2_tbt
		shift = shift_builder(x, S_arr, S2=false)
		#return cartan_tbt_l1_cost(tbt_so - shift, true)
		return cartan_tbt_l2_cost(tbt_so - shift, true)
	end

	x0 = zeros(5)

	sol = optimize(cost, x0, BFGS())

	#@show cost(x0)
	#@show sol.minimum
	x = sol.minimizer
	#shift = x[1]*Nα_tbt + x[2]*Nβ_tbt + x[3]*Nα2_tbt + x[4]*NαNβ_tbt + x[5]*Nβ2_tbt
	shift = shift_builder(x, S_arr, S2=false)

	return tbt_so - shift, sol.minimizer
end

function cartan_tbt_purification(tbt :: Tuple, spin_orb=true)
	# input: cartan tbt operator tuple (obt,tbt) (and whether it is in spin-orbitals or orbitals)
	# output: cartan two-body tensor operator in spin-orbitals with shifted symmetries, and shift constants
	if spin_orb == false
		tbt_so = obt_to_tbt(obt_orb_to_so(tbt[1])) + tbt_orb_to_so(tbt[2])
	else
		tbt_so = obt_to_tbt(tbt[1]) + tbt[2] 
	end

	return cartan_tbt_purification(tbt_so, true)
end

function symmetry_cuadratic_optimization(tbt :: Array, spin_orb=true; S2=true)
	println("Running symmetry reduction routine with S2=$S2")
	#finds optimal shift by minimizing cost of tbt
	#includes Nα, Nβ, Nα², Nα*Nβ, Nβ², and S² operators for symmetries
	if spin_orb == false
		tbt_so = tbt_orb_to_so(tbt)
	else
		tbt_so = tbt
	end
	n_qubit = size(tbt_so)[1]

	S_arr = casimirs_builder(n_qubit, S2=S2)

	s_len = length(S_arr)

	A_mat = zeros(s_len,s_len)
	v_vec = zeros(s_len)

	for i in 1:s_len
		v_vec[i] = sum(tbt_so .* S_arr[i])
	end

	for i in 1:s_len
		for j in i:s_len
			A_mat[i,j] = sum(S_arr[i] .* S_arr[j])
			A_mat[j,i] = A_mat[i,j]
		end
	end

	A_inv = inv(A_mat)

	x_vec = A_inv * v_vec

	tbt_sym = copy(tbt_so)
	for i in 1:s_len
		tbt_sym -= x_vec[i] * S_arr[i]
	end

	return tbt_sym, x_vec
end