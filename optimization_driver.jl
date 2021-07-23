#FUNCTIONS FOR OPTIMIZATION DRIVERS (E.G. FULL-RANK, GREEDY)
function greedy_driver(target, decomp_tol; reps = 1, 
	α_max=500, grad=true, verbose=true, spin_orb = true, 
	frag_flavour = META.ff, u_flavour=META.uf)

	n = length(target[:,1,1,1])
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	xsize = u_num + fcl

	tot_reps = reps * number_of_classes(frag_flavour)

	K_dict = zeros(Int64, α_max)
	K_map = zeros(Int64, tot_reps)

	FRAGS = fragment[]
	sizehint!(FRAGS, α_max)
	
	#initialize map
	idx = 1
	for i1 in 1:number_of_classes(frag_flavour)
		for i2 in 1:reps
			K_map[idx] = i1
			idx += 1
		end
	end

	cost = tbt_cost(0, target)
	curr_tbt = 0 .* target

	α = 1
	FCost = SharedArray(zeros(tot_reps))
	X_ARR = SharedArray(zeros(tot_reps, xsize))
	println("Starting two-body tensor cost is $cost")
	while cost > decomp_tol && α <= α_max
		println("Starting parallel calculation for fragment number $α")
		t00 = time()
		@sync @distributed for idx in 1:tot_reps
			k = K_map[idx]
			sol = greedy_step_optimization(target-curr_tbt, k, frag_flavour=frag_flavour,
								 u_flavour=u_flavour, grad=grad, spin_orb=spin_orb, n = n)
			FCost[idx] = sol.minimum
			X_ARR[idx,:] .= sol.minimizer
		end
		
		ind_min = sortperm(FCost)[1]
		k_min = K_map[ind_min]
		K_dict[α] = k_min
		cost = FCost[ind_min]
		x0 = X_ARR[ind_min,:]
		frag = fragment(x0[fcl+1:end], x0[1:fcl], k_min, n_qubit, spin_orb)
		curr_tbt += fragment_to_tbt(frag)
		push!(FRAGS, frag)
		println("Finished calculation for fragment $α after $(time()-t00) seconds,
		 chose transformation of type $k_min, current cost is $cost")
		if verbose == true
			println("X vector for fragment $α is:")
			@show x0
		end
		α += 1
	end

	α -= 1
	if cost < decomp_tol
		println("Finished succesfully using $α fragments, final cost is $cost")
	else
		println("Calculation not converged after $α fragments, current cost is $cost")
	end

	println("Finished greedy optimization with transformation train $(K_dict[1:α])")
	return FRAGS
end


function full_rank_driver(target, decomp_tol; reps = 1, 
	α_max=500, grad=true, verbose=true, x0=Float64[], 
	K0=Int64[], spin_orb = true, pre_optimize=PRE_OPT, 
	frag_flavour = META.ff, u_flavour=META.uf)

	println("Starting full_rank_driver with pre_optimization = $PRE_OPT")

	n = length(target[:,1,1,1])
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	xsize = u_num + fcl

	tot_reps = reps * number_of_classes(frag_flavour)

	K_dict = zeros(Int64,α_max)
	K_map = zeros(Int64, tot_reps)
	
	#initialize map
	idx = 1
	for i1 in 1:number_of_classes(frag_flavour)
		for i2 in 1:reps
			K_map[idx] = i1
			idx += 1
		end
	end

	ini_length = length(K0)
	if ini_length != 0
		println("Starting conditions found for full rank optimization,
		 using x0 parameters and K0 tensor train for $ini_length fragments")
		@show x0
		@show K0
		K_dict[1:ini_length] .= K0
		cost = full_rank_cost(x0, target, K0, spin_orb = spin_orb)
	else
		cost = tbt_cost(0, target)
	end
	
	α = ini_length + 1
	FCost = SharedArray(zeros(tot_reps))
	X_ARR = SharedArray(zeros(tot_reps, α_max*xsize))
	@show cost
	while cost > decomp_tol && α <= α_max
		x_len = length(x0) + xsize
		println("Starting parallel calculation for fragment number $α")
		t00 = time()
		@sync @distributed for idx in 1:tot_reps
			k = K_map[idx]
			class_arr = vcat(K_dict[1:α-1]..., k)
			#pre-optimization routine, first run greedy step and use as starting conditions for full-rank
			if pre_optimize == false
				#starting initial random condition, initialize frag.cn to zeros and u_params to [0,2π]
				xrand = zeros(xsize)
				xrand[2:end] = 2π * rand(u_num+fcl-1)
				x0_frag = vcat(x0..., xrand...)
			else
				#println("Starting pre-optimizing routine with cost $cost")
				t01 = time()	
				tbt_curr = copy(target)
				x_ini = 1
				for frag_num in 1:α-1
					xcur = x0[x_ini:x_ini+xsize-1]
					frag = fragment(xcur[fcl+1:end], xcur[1:fcl], class_arr[frag_num], n_qubit, spin_orb)
					tbt_curr -= fragment_to_tbt(frag)
					x_ini += xsize
				end
				sol = greedy_step_optimization(tbt_curr, k, spin_orb = spin_orb,
								 grad=grad, n = n, frag_flavour=frag_flavour, u_flavour=u_flavour)
				xsol = sol.minimizer
				x0_frag = vcat(x0..., xsol...)
				#println("Cost after 1-step optimization is $(sol.minimum), time spent was $(time()-t01) seconds")
			end

			sol = full_rank_optimization(target, class_arr, frag_flavour=frag_flavour,
							 u_flavour=u_flavour, x0=x0_frag, grad=grad, spin_orb=spin_orb, n = n)
			FCost[idx] = sol.minimum
			X_ARR[idx,1:x_len] .= sol.minimizer
		end
		
		ind_min = sortperm(FCost)[1]
		k_min = K_map[ind_min]
		K_dict[α] = k_min
		cost = FCost[ind_min]
		x0 = X_ARR[ind_min, 1:x_len]
		println("Finished calculation for fragment $α after $(time()-t00) seconds, chose 
			transformation of type $k_min, current cost is $cost")
		if verbose == true
			K0 = K_dict[1:α]
			println("Using transformation train K0 and parameters x0:")
			@show K0
			@show x0
		end
		α += 1
	end

	α -= 1
	if cost < decomp_tol
		println("Finished succesfully using $α fragments, final cost is $cost")
	else
		println("Calculation not converged after $α fragments, current cost is $cost")
	end

	FRAGS = fragment[]
	sizehint!(FRAGS, α)
	for frag_num in 1:α
		ini_ind = xsize * (frag_num-1) + 1
		x = x0[ini_ind:ini_ind+xsize-1]
		frag = fragment(x[fcl+1:end], x[1:fcl], K_dict[frag_num], n_qubit, spin_orb)
		push!(FRAGS, frag)
	end

	return FRAGS
end

function full_rank_non_iterative_driver(target; 
	grad=true, verbose=true, x0=Float64[], 
	K0=Int64[], spin_orb = true, 
	frag_flavour = META.ff, u_flavour=META.uf)

	println("Starting full_rank_non_iterative_driver")

	n = length(target[:,1,1,1])
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	xsize = u_num + fcl

	α = length(K0)
	
	@time sol = full_rank_optimization(target, K0, frag_flavour=frag_flavour,
							 u_flavour=u_flavour, x0=x0, grad=grad, spin_orb=spin_orb, n=n)
	x0 = sol.minimizer
	cost = sol.minimum
	println("Final cost is $cost")
	if verbose == true
		@show K0
		@show x0
	end
	
	FRAGS = fragment[]
	sizehint!(FRAGS, α)
	for frag_num in 1:α
		ini_ind = xsize * (frag_num-1) + 1
		x = x0[ini_ind:ini_ind+xsize-1]
		frag = fragment(x[fcl+1:end], x[1:fcl], K0[frag_num], n_qubit, spin_orb)
		push!(FRAGS, frag)
	end

	return FRAGS
end

function orthogonal_greedy_driver(target, decomp_tol; reps = 1, 
	α_max=500, grad=true, verbose=true, spin_orb = true, 
	frag_flavour = META.ff, u_flavour=META.uf, λ = λort)

	n = length(target[:,1,1,1])
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	xsize = u_num + fcl

	tot_reps = reps * number_of_classes(frag_flavour)

	K_dict = zeros(Int64, α_max)
	K_map = zeros(Int64, tot_reps)

	FRAGS = fragment[]
	sizehint!(FRAGS, α_max)
	
	#initialize map
	idx = 1
	for i1 in 1:number_of_classes(frag_flavour)
		for i2 in 1:reps
			K_map[idx] = i1
			idx += 1
		end
	end

	cost = tbt_cost(0, target)

	α = 1
	FCost = SharedArray(zeros(tot_reps))
	X_ARR = SharedArray(zeros(tot_reps, xsize))
	COEFFS_ARR = SharedArray(zeros(tot_reps, α_max))
	println("Starting two-body tensor cost is $cost")
	while cost > decomp_tol && α <= α_max
		println("Starting parallel calculation for fragment number $α")
		t00 = time()
		@sync @distributed for idx in 1:tot_reps
			k = K_map[idx]
			sol = orthogonal_greedy_step_optimization(target, k, FRAGS, frag_flavour=frag_flavour,
								 u_flavour=u_flavour, grad=grad, spin_orb=spin_orb, n=n, λ=λ)
			FCost[idx] = sol.minimum
			X_ARR[idx,:] .= sol.minimizer[1:xsize]
			COEFFS_ARR[idx,1:α-1] = sol.minimizer[xsize+1:end]
		end
		
		ind_min = sortperm(FCost)[1]
		k_min = K_map[ind_min]
		K_dict[α] = k_min
		cost = FCost[ind_min]
		x0 = X_ARR[ind_min,:]
		frag = fragment(x0[fcl+1:end], x0[1:fcl], k_min, n_qubit, spin_orb)
		push!(FRAGS, frag)
		#updating coefficients in frags...
		for (i,frag) in enumerate(FRAGS)
			frag.cn[1] = COEFFS_ARR[ind_min, i]
		end

		println("Finished calculation for fragment $α after $(time()-t00) seconds,
		 chose transformation of type $k_min, current cost is $cost")
		if verbose == true
			println("X vector for fragment $α is:")
			@show x0
		end
		α += 1
	end

	α -= 1
	if cost < decomp_tol
		println("Finished succesfully using $α fragments, final cost is $cost")
	else
		println("Calculation not converged after $α fragments, current cost is $cost")
	end

	println("Finished greedy optimization with transformation train $(K_dict[1:α])")
	return FRAGS
end