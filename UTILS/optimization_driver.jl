function run_optimization(opt_flavour, tbt, decomp_tol,
			 reps, α_max, grad, verbose, x0, K0, spin_orb, f_name;
			 frag_flavour=META.ff, u_flavour=META.uf)

	if opt_flavour == "full-rank" || opt_flavour == "fr"
		@time FRAGS = full_rank_driver(tbt, decomp_tol,
		 	f_name = f_name, reps = reps, α_max=α_max, grad=grad,
		 	verbose=verbose, x0=x0, K0=K0, spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)

	elseif opt_flavour == "greedy" || opt_flavour == "g"
		@time FRAGS = greedy_driver(tbt, decomp_tol, f_name = f_name,
				 reps = reps, α_max=α_max, grad=grad, verbose=verbose, 
				 spin_orb=spin_orb, x0=x0, K0=K0, frag_flavour=frag_flavour, u_flavour=u_flavour)

	elseif opt_flavour == "relaxed-greedy" || opt_flavour == "rg"
		@time FRAGS = relaxed_greedy_driver(tbt, decomp_tol, f_name = f_name,
				reps = reps, α_max=α_max, grad=grad, verbose=verbose,
				spin_orb=spin_orb, x0=x0, K0=K0, frag_flavour=frag_flavour, u_flavour=u_flavour)

	elseif opt_flavour == "og" || opt_flavour == "orthogonal-greedy"
		println("Using λ=$λort for orthogonal greedy constrain value")
		@time FRAGS = orthogonal_greedy_driver(tbt, decomp_tol, f_name = f_name,
		reps = reps, α_max=α_max, grad=grad, verbose=verbose, spin_orb=spin_orb, 
		frag_flavour=frag_flavour, u_flavour=u_flavour)

	elseif opt_flavour == "frni" || opt_flavour == "full-rank-non-iterative"
		num_classes = number_of_classes(frag_flavour)
		class_train = rand(1:num_classes, α_max)
		@show class_train
		@time FRAGS = full_rank_non_iterative_driver(tbt, f_name=f_name, grad=grad, verbose=verbose,
				x0=x0, K0=class_train, spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
		
	else
		error("Trying to do decomposition with optimization flavour $opt_flavour, not implemented!")
	end

	if CSA_family(frag_flavour) == false
		println("Showing L1 norm of sum of coefficient fragments")
		@show sum(abs.([frag.cn[1] for frag in FRAGS]))
	end

	return FRAGS
end

#FUNCTIONS FOR OPTIMIZATION DRIVERS (E.G. FULL-RANK, GREEDY)
function greedy_driver(target, decomp_tol; reps = 1, 
	α_max=500, grad=true, verbose=true, spin_orb = true, 
	frag_flavour = META.ff, u_flavour=META.uf,
	x0 = Float64[], K0 = Int64[], saving=saving,
	f_name = NAME)
	
	if typeof(target) <: Tuple
		if singles_family(frag_flavour)
			n = size(target[1])[1]
		else
			error("Got Tuple for target tensors, but frag_flavour method $frag_flavour does not include singles!")
		end
	else
		n = size(target)[1]
	end

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

	if length(K0) != 0
		println("Initial conditions found")
		#@show x0
		#@show K0
		if length(K0) <= α_max
			max_frags = length(K0)
		else
			max_frags = α_max
		end
		α = max_frags + 1
		K_dict[1:max_frags] .= K0[1:max_frags]
		idx = 1
		for i in 1:max_frags
			xeff = x0[idx:idx+xsize-1]
			frag = fragment(xeff[fcl+1:end], xeff[1:fcl], K0[i], n_qubit, spin_orb)
			push!(FRAGS, frag)
			target -= fragment_to_tbt(frag)
			idx += xsize
		end
	else
		α = 1
	end
	α_ini = copy(α)

	cost = tbt_cost(0, target)
	curr_tbt = 0 .* target

	x0_tot = copy(x0)
	K0_tot = copy(K0)
	FCost = SharedArray(zeros(tot_reps))
	X_ARR = SharedArray(zeros(tot_reps, xsize))
	println("Starting two-body tensor cost is $cost")
	while cost > decomp_tol && α <= α_max
		println("Starting parallel calculation for fragment number $α")
		t00 = time()
		@sync @distributed for idx in 1:tot_reps
			k = K_map[idx]
			sol = greedy_step_optimization(target .- curr_tbt, k, frag_flavour=frag_flavour,
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
		curr_tbt += fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
		push!(FRAGS, frag)
		println("Finished calculation for fragment $α after $(time()-t00) seconds,
		 chose transformation of type $k_min, current cost is $cost")
		if verbose == true
			println("X vector for fragment $α is:")
			@show x0
		end
		α += 1
		if saving == true
			append!(x0_tot,x0)
			push!(K0_tot,k_min)
			overwrite_xK(f_name,x0_tot,K0_tot)
		end
	end

	if saving == true
		overwrite_xK(f_name,x0_tot,K0_tot)
	end

	α -= 1
	if cost < decomp_tol
		println("Finished succesfully using $α fragments, final cost is $cost")
	else
		println("Calculation not converged after $α fragments, current cost is $cost")
	end

	println("Finished greedy optimization with transformation train $(K_dict[1:α])")
	@show length(FRAGS)
	return FRAGS
end

function relaxed_greedy_driver(target, decomp_tol; reps = 1, 
	α_max=500, grad=true, verbose=true, spin_orb = true, 
	frag_flavour = META.ff, u_flavour=META.uf,
	x0 = Float64[], K0 = Int64[], saving=saving,
	f_name = NAME)
	
	if typeof(target) <: Tuple
		println("Target is tuple, decomposing singles and doubles tensors")
		if singles_family(frag_flavour)
			n = size(target[1])[1]
		else
			error("Got Tuple for target tensors, but frag_flavour method $frag_flavour does not include singles!")
		end
	else
		n = size(target)[1]
	end

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

	U_PREV = SharedArray(zeros(u_num, α_max))
	X_PREV = SharedArray(zeros(fcl, α_max))
	curr_tbt = target .* 0

	if length(K0) != 0
		println("Initial conditions found")
		#@show x0
		#@show K0
		if length(K0) <= α_max
			max_frags = length(K0)
		else
			max_frags = α_max
		end
		α = max_frags + 1
		K_dict[1:max_frags] .= K0[1:max_frags]
		idx = 1
		for i in 1:max_frags
			xeff = x0[idx:idx+xsize-1]
			frag = fragment(xeff[fcl+1:end], xeff[1:fcl], K0[i], n_qubit, spin_orb)
			X_PREV[:, i] = xeff[2:fcl]
			U_PREV[:, i] = xeff[fcl+1:end]
			push!(FRAGS, frag)
			curr_tbt -= fragment_to_tbt(frag)
			idx += xsize
		end
	else
		α = 1
	end

	cost = tbt_cost(curr_tbt, target)
	α_ini = copy(α)
	
	x0_tot = copy(x0)
	K0_tot = copy(K0)
	FCost = SharedArray(zeros(tot_reps))
	X_ARR = SharedArray(zeros(tot_reps, xsize + α_max - 1))
	println("Starting two-body tensor cost is $cost")
	while cost > decomp_tol && α <= α_max
		println("Starting parallel calculation for fragment number $α")
		t00 = time()
		@sync @distributed for idx in 1:tot_reps
			k = K_map[idx]
			K0_eff = cat(K0_tot, k, dims=1)
			sol = relaxed_greedy_optimization(target, K0_eff, U_PREV[:, 1:α-1], X_PREV[:, 1:α-1]; x0=false,
			 grad=grad, spin_orb=spin_orb, n = n, frag_flavour=frag_flavour, u_flavour=u_flavour)
			FCost[idx] = sol.minimum
			X_ARR[idx,1:α-1+xsize] .= sol.minimizer
		end
		
		ind_min = sortperm(FCost)[1]
		k_min = K_map[ind_min]
		K_dict[α] = k_min
		cost = FCost[ind_min]
		x0 = X_ARR[ind_min,α:α+xsize-1]
		frag = fragment(x0[fcl+1:end], x0[1:fcl], k_min, n_qubit, spin_orb)
		#update coefficients and build current tbt...
		curr_tbt = fragment_to_tbt(frag)
		for i in 1:α-1
			FRAGS[i].cn[1] = X_ARR[ind_min, i]
			X_PREV[1, i] = X_ARR[ind_min, i]
			curr_tbt += fragment_to_tbt(FRAGS[i])
		end
		X_PREV[:, α] = x0[1:fcl]
		U_PREV[:, α] = x0[fcl+1:end]

		push!(FRAGS, frag)
		println("Finished calculation for fragment $α after $(time()-t00) seconds,
		 chose transformation of type $k_min, current cost is $cost")
		if verbose == true
			println("X vector for fragment $α is:")
			@show x0
		end
		α += 1
		if saving == true
			append!(x0_tot,x0)
			push!(K0_tot,k_min)
			overwrite_xK(f_name,x0_tot,K0_tot)
		end
	end

	if saving == true
		overwrite_xK(f_name,x0_tot,K0_tot)
	end

	α -= 1
	if cost < decomp_tol
		println("Finished succesfully using $α fragments, final cost is $cost")
	else
		println("Calculation not converged after $α fragments, current cost is $cost")
	end

	println("Finished greedy optimization with transformation train $(K_dict[1:α])")
	@show length(FRAGS)
	return FRAGS
end

function block_size_calculator(b_size, num_classes)
	if num_classes == 0 || b_size == 0
		return 0
	end

	if b_size == 1
		return num_classes
	end

	if num_classes == 1
		return 1
	end

	S = 0
	for i in num_classes:-1:1
		S += block_size_calculator(b_size-1, i)
	end

	return S
end

function append_vec(v, A)
	# returns [v A] matrix, which is matrix A with appended vector A on the left-most coordinate
	sA = size(A)
	sv = length(v)

	if sA[1] != sv
		error("Dimensions mismatch!")
	end

	Aaug = zeros(typeof(A[1]), sA[1], sA[2]+1)
	Aaug[:,1] = v
	Aaug[:,2:end] = A

	return Aaug
end

function append_num(n, A)
	v = n * ones(size(A)[1])

	return append_vec(v, A)
end

function block_map(b_size, num_classes)
	if b_size == 1
		K_map = zeros(Int64, num_classes, 1)
		K_map .= collect(num_classes:-1:1)
		return K_map
	end

	K_map = zeros(Int64, block_size_calculator(b_size, num_classes), b_size)
	idx = 1
	for i in num_classes:-1:1
		K_inst = block_map(b_size-1, i)
		K_inst = append_num(i, K_inst)
		k_size = size(K_inst)[1] 
		K_map[idx:idx+k_size-1,:] = K_inst
		idx += k_size
	end

	return K_map
end

function relaxed_block_greedy_driver(target, decomp_tol; reps = 1, 
	α_max=500, grad=true, verbose=true, spin_orb = true, 
	frag_flavour = META.ff, u_flavour=META.uf,
	x0 = Float64[], K0 = Int64[], saving=saving,
	f_name = NAME, b_size = block_size)
	
	if typeof(target) <: Tuple
		if singles_family(frag_flavour)
			n = size(target[1])[1]
		else
			error("Got Tuple for target tensors, but frag_flavour method $frag_flavour does not include singles!")
		end
	else
		n = size(target)[1]
	end

	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	xsize = u_num + fcl

	num_classes = number_of_classes(frag_flavour)
	K_map = block_map(b_size, num_classes)
	# K_map[i,:] contains b_size array of classes for optimization
	tot_reps = reps * size(K_map)[1]

	K_dict = zeros(Int64, α_max)

	FRAGS = fragment[]
	sizehint!(FRAGS, α_max)

	U_PREV = SharedArray(zeros(u_num, tot_reps))
	X_PREV = SharedArray(zeros(fcl, tot_reps))
	curr_tbt = 0 .* target

	if length(K0) != 0
		curr_tbt = target .* 0
		println("Initial conditions found")
		#@show x0
		#@show K0
		if length(K0) <= α_max
			max_frags = length(K0)
		else
			max_frags = α_max
		end
		α = max_frags + 1
		K_dict[1:max_frags] .= K0[1:max_frags]
		idx = 1
		for i in 1:max_frags
			xeff = x0[idx:idx+xsize-1]
			frag = fragment(xeff[fcl+1:end], xeff[1:fcl], K0[i], n_qubit, spin_orb)
			X_PREV[:, i] = xeff[2:fcl]
			U_PREV[:, i] = xeff[fcl+1:end]
			push!(FRAGS, frag)
			curr_tbt -= fragment_to_tbt(frag)
			idx += xsize
		end
	else
		α = 1
	end

	cost = tbt_cost(curr_tbt, target)
	α_ini = copy(α)
	
	x0_tot = copy(x0)
	K0_tot = copy(K0)
	FCost = SharedArray(zeros(tot_reps))
	X_ARR = SharedArray(zeros(tot_reps, xsize + tot_reps - 1))
	println("Starting two-body tensor cost is $cost")
	while cost > decomp_tol && α <= α_max
		println("Starting block-parallel calculation for fragment number $α")
		t00 = time()
		@sync @distributed for idx in 1:tot_reps
			k = K_map[idx,:]
			K0_eff = cat(K0_tot, k, dims=1)
			sol = relaxed_block_greedy_optimization(target, K0_eff, U_PREV[:, 1:α-1], X_PREV[:, 1:α-1]; b_size=b_size,
			 x0=false, grad=grad, spin_orb=spin_orb, n=n, frag_flavour=frag_flavour, u_flavour=u_flavour)
			FCost[idx] = sol.minimum
			X_ARR[idx,1:α-1+xsize*b_size] .= sol.minimizer
		end
		
		ind_min = sortperm(FCost)[1]
		k_min = K_map[ind_min,:]
		K_dict[α:α+b_size-1] = k_min
		cost = FCost[ind_min]
		x0 = X_ARR[ind_min,α:α-1+xsize*b_size]

		curr_tbt = 0 .* target
		x_ini = 1
		for i in 1:b_size
			x0 = x[x_ini:x_ini+x_size-1]
			X_PREV[:, α-1+i] = x0[1:fcl]
			U_PREV[:, α-1+i] = x0[fcl+1:end]
			frag = fragment(x0[fcl+1:end], x0[1:fcl], class_arr[end], n_qubit, spin_orb)
			push!(FRAGS, frag)
			curr_tbt += fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
			x_ini += x_size
		end
		
		for i in 1:α-1
			FRAGS[i].cn[1] = X_ARR[ind_min, i]
			X_PREV[1, i] = X_ARR[ind_min, i]
			curr_tbt += fragment_to_tbt(FRAGS[i])
		end

		println("Finished calculation for fragment block $α-$(α+b_size-1) after $(time()-t00) seconds,
		 chose transformation train $k_min, current cost is $cost")
		if verbose == true
			println("X vector for fragment $α is:")
			@show x0
		end
		α += b_size
		if saving == true
			append!(x0_tot,x0)
			append!(K0_tot,k_min)
			overwrite_xK(f_name,x0_tot,K0_tot)
		end
	end

	if saving == true
		overwrite_xK(f_name,x0,K0)
	end

	α -= 1
	if cost < decomp_tol
		println("Finished succesfully using $α fragments, final cost is $cost")
	else
		println("Calculation not converged after $α fragments, current cost is $cost")
	end

	println("Finished greedy optimization with transformation train $(K_dict[1:α])")
	@show length(FRAGS)
	return FRAGS
end

function full_rank_driver(target, decomp_tol; reps = 1, 
	α_max=500, grad=true, verbose=true, x0=Float64[], 
	K0=Int64[], spin_orb = true, pre_optimize=PRE_OPT, 
	frag_flavour = META.ff, u_flavour=META.uf,
	saving=saving, f_name = NAME)

	#println("Starting full_rank_driver with pre_optimization = $PRE_OPT")

	if typeof(target) <: Tuple
		if singles_family(frag_flavour)
			n = size(target[1])[1]
		else
			error("Got Tuple for target tensors, but frag_flavour method $frag_flavour does not include singles!")
		end
	else
		n = size(target)[1]
	end

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
		 using $ini_length fragments")
		#@show x0
		#@show K0
		K_dict[1:ini_length] .= K0
		cost = full_rank_cost(x0, target, K0, spin_orb = spin_orb, n=n)
	else
		cost = tbt_cost(0, target)
	end
	
	α = ini_length + 1
	FCost = SharedArray(zeros(tot_reps))
	X_ARR = SharedArray(zeros(tot_reps, α_max*xsize))
	println("Starting optimization, current cost is $cost")
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
		K0 = K_dict[1:α]
		if verbose == true
			println("Using transformation train K0 and parameters x0:")
			@show K0
			@show x0
		end
		α += 1
		if saving == true
			overwrite_xK(f_name,x0,K0)
		end
	end

	if saving == true
		overwrite_xK(f_name,x0,K0)
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
	frag_flavour = META.ff, u_flavour=META.uf,
	saving=saving, f_name=NAME)

	println("Starting full_rank_non_iterative_driver")

	if typeof(target) <: Tuple
		if singles_family(frag_flavour)
			n = size(target[1])[1]
		else
			error("Got Tuple for target tensors, but frag_flavour method $frag_flavour does not include singles!")
		end
	else
		n = size(target)[1]
	end

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

	if saving == true
		overwrite_xK(f_name, x0, K0)
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
	frag_flavour = META.ff, u_flavour=META.uf, λ = λort,
	saving=saving, f_name=NAME)

	if typeof(target) <: Tuple
		if singles_family(frag_flavour)
			n = size(target[1])[1]
		else
			error("Got Tuple for target tensors, but frag_flavour method $frag_flavour does not include singles!")
		end
	else
		n = size(target)[1]
	end

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
	Karr = Int64[]
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
		push!(Karr, k_min)
		K_dict[α] = k_min
		cost = FCost[ind_min]
		global x0 = X_ARR[ind_min,:]
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
	if saving == true
		overwrite_xK(f_name,x0,Karr)
	end

	println("Finished greedy optimization with transformation train $(K_dict[1:α])")
	return FRAGS
end

function in_block(opt_flavour)
	block_list = ["bg", "block-greedy", "rbg", "relaxed-block-greedy"]
	return opt_flavour in block_list
end