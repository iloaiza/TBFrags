function majorana_l1_cost(tbt_mo :: Tuple, n = size(tbt_mo[1])[1])
	obt_mod = tbt_mo[1] + 2*sum([tbt_mo[2][:,:,r,r] for r in 1:n])# - sum([tbt_mo[2][:,r,r,:] for r in 1:n]) already included in chemist notation
	
	λT = sum(abs.(obt_mod))
	global λV = 0.5 * sum(abs.(tbt_mo[2]))

	for r in 1:n
		for p in r+1:n
			for q in 1:n
				for s in q+1:n
					global λV += abs(tbt_mo[2][p,q,r,s] - tbt_mo[2][p,s,r,q])
				end
			end
		end
	end

	return λT+λV
end

function majorana_U_optimizer(tbt_mo :: Tuple, u_flavour=MF_real())
	n = size(tbt_mo[1])[1]
	u_num = unitary_parameter_number(n, u_flavour)

	u_params = zeros(u_num)

	function cost(x)
		obt_rot, tbt_rot = generalized_unitary_rotation(x, tbt_mo, n, u_flavour)
		return majorana_l1_cost((obt_rot, tbt_rot), n)
	end
	ini_cost = cost(u_params)
	u_params = 2π*rand(u_num)

	@show ini_cost
	sol = optimize(cost, u_params, BFGS())
	@show sol.minimum

	obt_rot, tbt_rot = generalized_unitary_rotation(sol.minimizer, tbt_mo, n, u_flavour)

	return (obt_rot, tbt_rot)
end

function majorana_parallel_U_optimizer(tbt_mo :: Tuple, reps=10, u_flavour=MF_real())
	n = size(tbt_mo[1])[1]
	u_num = unitary_parameter_number(n, u_flavour)

	u_params = SharedArray(zeros(u_num, reps))

	function cost(x)
		obt_rot, tbt_rot = generalized_unitary_rotation(x, tbt_mo, n, u_flavour)
		return majorana_l1_cost((obt_rot, tbt_rot), n)
	end
	ini_cost = cost(u_params)
	u_params .= 2π*rand(u_num, reps)

	FCosts = SharedArray(zeros(reps))
	@show ini_cost
	@sync @distributed for i in 1:reps
		sol = optimize(cost, u_params[:,i], BFGS())
		u_params[:,i] = sol.minimizer
		FCosts[i] = sol.minimum
	end

	indperm = sortperm(FCosts)
	@show FCosts[indperm[1]]
	@show FCosts[indperm[end]]
	opt_params = u_params[:, indperm[1]]

	obt_rot, tbt_rot = generalized_unitary_rotation(opt_params, tbt_mo, n, u_flavour)

	return (obt_rot, tbt_rot)
end

function shifted_majorana_λV_cost(tbt_mo :: Tuple, s_vec, n = size(tbt_mo[1])[1])
	# s_vec ≡ [s(Nα²/Nβ²), s(NαNβ)], corresponding to are shift coefficients
	tbt_αβ = copy(tbt_mo[2])	 # tbt_αβ = tbt_βα
	tbt_α = copy(tbt_αβ)	# tbt_α = tbt_β == tbt_αα

	for i in 1:n
		for j in 1:n
			tbt_αβ[i,i,j,j] -= s_vec[2]
			tbt_α[i,i,j,j] -= s_vec[1]
		end
	end

	global λV = sum(abs.(tbt_αβ))

	for r in 1:n
		for p in r+1:n
			for q in 1:n
				for s in q+1:n
					global λV += abs(tbt_α[p,q,r,s] - tbt_α[p,s,r,q])
				end
			end
		end
	end

	λV *= 0.5

	obt_mod = tbt_mo[1] + 2*sum([tbt_mo[2][:,:,r,r] for r in 1:n])# - sum([tbt_mo[2][:,r,r,:] for r in 1:n]) already included in chemist notation
	obt_diag = diag(obt_mod)
	global λT = sum(abs.(obt_mod - Diagonal(obt_mod)))
	for i in 1:n
		global λT += abs(obt_diag[i] - n*(1+1im)*sum(s_vec))
	end

	@show λT, λV

	return λV + λT
end

function post_shift_majorana_optimization(tbt_mo :: Tuple, s_vec, reps=10, u_flavour=MF_real())
	# s_vec ≡ [s(Nα²/Nβ²), s(NαNβ)], corresponding to are shift coefficients

	n = size(tbt_mo[1])[1]
	u_num = unitary_parameter_number(n, u_flavour)

	u_params = SharedArray(zeros(u_num, reps))

	function cost(x)
		obt_rot, tbt_rot = generalized_unitary_rotation(x, tbt_mo, n, u_flavour)
		return shifted_majorana_λV_cost((obt_rot, tbt_rot), s_vec, n)
	end

	ini_cost = cost(u_params)
	u_params .= 2π*rand(u_num, reps)

	FCosts = SharedArray(zeros(reps))
	@show ini_cost
	@sync @distributed for i in 1:reps
		sol = optimize(cost, u_params[:,i], BFGS())
		u_params[:,i] = sol.minimizer
		FCosts[i] = sol.minimum
	end

	indperm = sortperm(FCosts)
	@show FCosts[indperm[1]]
	@show FCosts[indperm[end]]
	opt_params = u_params[:, indperm[1]]

	obt_rot, tbt_rot = generalized_unitary_rotation(opt_params, tbt_mo, n, u_flavour)

	return (obt_rot, tbt_rot)
end