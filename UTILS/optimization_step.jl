#FUNCTIONS WHICH HAVE SINGLE STEP USED IN OPTIMIZATION DRIVERS

function greedy_step_optimization(target, class; spin_orb = false,
				 grad=false, x0=false, n = length(target[:,1,1,1]), frag_flavour=META.ff, u_flavour=META.uf)
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	num_zeros = frag_num_zeros(n, frag_flavour)

	if x0 == false
		#starting initial random condition, initialize frag.cn to zeros and u_params to [0,2π]
		x0 = zeros(u_num + fcl)
		x0[1+num_zeros:end] = 2π*rand(fcl+u_num-num_zeros)
	end

	function cost(x)
		return parameter_cost(x, target, class, n=n,
				 spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end

	if grad == false
		return optimize(cost, x0, BFGS())
	else
		function grad!(storage, x)
			return parameter_gradient!(storage, x, target, class, n,
					 spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
		end
		return optimize(cost, grad!, x0, BFGS())
	end
end

function full_rank_optimization(target, class_arr; x0=false, grad=false,
				 spin_orb=false, n = length(target[:,1,1,1]), frag_flavour=META.ff, u_flavour=META.uf)
	num_frags = length(class_arr)

	function cost(x)
		return full_rank_cost(x, target, class_arr, n=n, spin_orb = spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end

	u_num = unitary_parameter_number(n, u_flavour)
	fcl = frag_coeff_length(n, frag_flavour)
	num_zeros = frag_num_zeros(n, frag_flavour)

	if x0 == false || x0 == []
		#starting initial random condition, initialize frag.cn[1] to zeros, and frag.cn[2:end] (if exist) and u_params to [0,2π]
		x0 = zeros(u_num+fcl, num_frags)
		x0[1+num_zeros:end,:] = 2π * rand(u_num+fcl-num_zeros, num_frags)
		x0 = vcat(x0...)
	end

	if grad == false
		return optimize(cost, x0, BFGS())
	else
		function grad!(storage, x)
			return full_rank_gradient!(storage, x, target, class_arr, n,
					 spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
		end
		return optimize(cost, grad!, x0, BFGS())
	end
end

function relaxed_greedy_optimization(target, class_arr, u_prev=[], x_prev=[]; x0=false, grad=false,
				 spin_orb=false, n = length(target[:,1,1,1]), frag_flavour=META.ff, u_flavour=META.uf)
	num_frags = length(class_arr)
	#x_prev: array of previous x, dimensions are [fcl, num_frags-1]. Includes previous coefficients for first guess
	#u_arr: array of previous unitaries, dimensions are [u_num, num_frags-1]

	function cost(x)
		return relaxed_cost(x, target, class_arr, u_prev, x_prev[2:end, :]; 
			n=n, spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end

	u_num = unitary_parameter_number(n, u_flavour)
	fcl = frag_coeff_length(n, frag_flavour)
	num_zeros = frag_num_zeros(n, frag_flavour)

	
	if x0 == false
		#starting initial random condition for newest fragment, initialize frag.cn to zeros and u_params to [0,2π]
		x0 = zeros(u_num + fcl)
		x0[1+num_zeros:end] = 2π*rand(fcl+u_num-num_zeros)
	end
	
	x0 = cat(x_prev[1,:], x0, dims=1)

	if grad == false
		return optimize(cost, x0, BFGS())
	else
		error("Gradient optimization not defined for relaxed-greedy optimization!")
		#=####### This is a placeholder from full-rank gradient optimization
		function grad!(storage, x)
			return full_rank_gradient!(storage, x, target, class_arr, n,
					 spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
		end
		return optimize(cost, grad!, x0, BFGS())
		# =#
	end
end

function relaxed_block_greedy_optimization(target, class_arr, u_prev=[], x_prev=[]; b_size = block_size, x0=false, grad=false,
				 spin_orb=false, n = length(target[:,1,1,1]), frag_flavour=META.ff, u_flavour=META.uf)
	num_frags = length(class_arr)
	#x_prev: array of previous x, dimensions are [fcl, num_frags-b_size]. Includes previous coefficients for first guess
	#u_arr: array of previous unitaries, dimensions are [u_num, num_frags-b_size]

	function cost(x)
		return relaxed_block_cost(x, target, class_arr, u_prev, x_prev[2:end, :]; 
			b_size=b_size, n=n, spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end

	u_num = unitary_parameter_number(n, u_flavour)
	fcl = frag_coeff_length(n, frag_flavour)
	num_zeros = frag_num_zeros(n, frag_flavour)
	
	if x0 == false || x0 == []
		#starting initial random condition, initialize frag.cn[1] to zeros, and frag.cn[2:end] (if exist) and u_params to [0,2π]
		x0 = zeros(u_num+fcl, b_size)
		x0[1+num_zeros:end,:] = 2π * rand(u_num+fcl-num_zeros, num_frags)
		x0 = vcat(x0...)
	end
	
	x0 = cat(x_prev[1,:], x0, dims=1)

	if grad == false
		return optimize(cost, x0, BFGS())
	else
		error("Gradient optimization not defined for relaxed-block-greedy optimization!")
		#=####### This is a placeholder from full-rank gradient optimization
		function grad!(storage, x)
			return full_rank_gradient!(storage, x, target, class_arr, n,
					 spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
		end
		return optimize(cost, grad!, x0, BFGS())
		# =#
	end
end

function classes_normalizer(frag_flavour = META.ff)
	num_classes = number_of_classes(frag_flavour)
	NORMS = zeros(num_classes)

	for i in 1:num_classes
		if typeof(frag_flavour) == CGMFR
			TBT = cgmfr_tbt_builder(i, 4)
		elseif typeof(frag_flavour) == O3
			TBT = o3_tbt_builder(i, 4)
		else 
			error("Class normalizer not defined for $frag_flavour for calculating norm in orthogonal-greedy optimization")
		end
		NORMS[i] = tbt_cost(TBT, 0 .* TBT)
	end

	return NORMS
end

function hilbert_norm(tbt1, tbt2, class1, class2; frag_flavour = META.ff, NORMS=classes_normalizer(frag_flavour))
	tbt_n1 = tbt1 ./ sqrt(NORMS[class1])
	tbt_n2 = tbt2 ./ sqrt(NORMS[class2])

	diff = 0 .* tbt_n1
	@einsum diff[a,b,c,d] = tbt_n1[a,b,c,d] - tbt_n2[d,c,b,a]

	return sum(abs2.(diff))
end

if opt_flavour == "ogreedy" || opt_flavour == "orthogonal-greedy" #define norms for hilbert norm for orthogonal-greedy algorithm
	global NORMS = classes_normalizer()
end

function orthogonal_cost(x, x_frags, class, target, frag_flavour, u_flavour, frags_arr, λ, n)
	tbt = parameter_to_tbt(x, class, n, frag_flavour=frag_flavour, u_flavour=u_flavour)
	num_frags = length(frags_arr)
	TBTs = zeros(num_frags,n,n,n,n)
	for (i,frag) in enumerate(frags_arr)
		TBTs[i,:,:,:,:] = fragment_to_normalized_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end
	tbt_curr = zeros(n,n,n,n)
	@einsum tbt_curr[a,b,c,d] = x_frags[i] * TBTs[i,a,b,c,d]

	pcost = tbt_cost(tbt, target-tbt_curr)

	if λ == 0 #just return adaptative greedy (i.e. adjusts fragment parameters + previous_frags_coeffs)
		return pcost
	end

	ocost = 0.0

	for frag in frags_arr
		tbt_arr = fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
		ocost += hilbert_norm(tbt, tbt_arr, class, frag.class, frag_flavour=frag_flavour, NORMS=NORMS)
	end

	return pcost*(1+λ*ocost)
end

function orthogonal_greedy_step_optimization(target, class, frags_arr;
			 				spin_orb = false, grad=false, x0=false, n = length(target[:,1,1,1]),
			 				frag_flavour=META.ff, u_flavour=META.uf, λ=λort)
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	num_frags = length(frags_arr)
	num_zeros = frag_num_zeros(n, frag_flavour)

	if x0 == false
		#starting initial random condition, initialize frag.cn to zeros and u_params to [0,2π]
		x0 = zeros(u_num + fcl + (num_frags*num_zeros) )
		x0[1+num_zeros:u_num+fcl] = 2π*rand(fcl+u_num-num_zeros)
		zidx = 0
		for i in 1:num_frags
			x0[u_num+fcl+zidx:u_num+fcl+zidx+num_zeros] .= frags_arr[i].cn[1:num_zeros]
			zidx += num_zeros
		end
	end

	function cost(x)
		return orthogonal_cost(x[1:u_num+fcl], x[u_num+fcl+1:end], class, target, frag_flavour, u_flavour, frags_arr, λ, n)
	end



	if grad == false
		return optimize(cost, x0, BFGS())
	else
		error("No gradient implemented for orthogonal-greedy optimization!")
		function grad!(storage, x)
			return parameter_gradient!(storage, x, target, class, n,
						 spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
		end
		return optimize(cost, grad!, x0, BFGS())
	end
end

function greedy_step_SD_optimization(ob_target, tb_target, class; spin_orb = false,
				 grad=false, x0=false, n = size(ob_target)[1], frag_flavour=META.ff, u_flavour=META.uf)
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	num_zeros = frag_num_zeros(n, frag_flavour)

	if x0 == false
		#starting initial random condition, initialize frag.cn to zeros and u_params to [0,2π]
		x0 = zeros(u_num + fcl)
		x0[1+num_zeros:end] = 2π*rand(fcl+u_num-num_zeros)
	end

	function cost(x)
		return SD_parameter_cost(x, ob_target, tb_target, class, n=n,
				 spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end

	if grad == false
		return optimize(cost, x0, BFGS())
	else
		error("Gradient not implemented for SD optimization")
		#=
		function grad!(storage, x)
			return parameter_gradient!(storage, x, target, class, n,
					 spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)
		end
		return optimize(cost, grad!, x0, BFGS())
		# =#
	end
end