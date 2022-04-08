#reduces Cartan polynomials using Casimir operators
function casimirs_builder(n_qubit; debug=false)
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
		#S2 = of.s_squared_operator(n_orbs)
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
	
	return Nα_obt, Nβ_obt, Nα_tbt, Nβ_tbt, Nα2_tbt, NαNβ_tbt, Nβ2_tbt
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

	_, _, Nα_tbt, Nβ_tbt, Nα2_tbt, NαNβ_tbt, Nβ2_tbt = casimirs_builder(n_qubit)

	function cost(x) 
		shift = x[1]*Nα_tbt + x[2]*Nβ_tbt + x[3]*Nα2_tbt + x[4]*NαNβ_tbt + x[5]*Nβ2_tbt
		cartan_tbt_l1_cost(tbt_so - shift, true)
	end

	x0 = zeros(5)

	sol = optimize(cost, x0, BFGS())

	#@show cost(x0)
	#@show sol.minimum
	x = sol.minimizer
	shift = x[1]*Nα_tbt + x[2]*Nβ_tbt + x[3]*Nα2_tbt + x[4]*NαNβ_tbt + x[5]*Nβ2_tbt

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
