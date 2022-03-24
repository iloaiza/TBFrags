function CSASD_tot_decomp(h_ferm; reps = 1, 
	grad=false, verbose=true, spin_orb = true, 
	frag_flavour = CSASD(), u_flavour=MF_real(),
	x0 = Float64[], saving=saving,
	f_name = NAME, α_max = 1, decomp_tol = 1e-6)

	tbt = fermionic.get_chemist_tbt(h_ferm, spin_orb=spin_orb)
	ob_op = of_simplify(h_ferm - tbt_to_ferm(tbt, spin_orb))
	obt = fermionic.get_obt(ob_op, spin_orb=spin_orb)

	n = length(tbt[:,1,1,1])
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	u_num = unitary_parameter_number(n)
	fcl = frag_coeff_length(n, frag_flavour)
	xsize = u_num + fcl

	tot_reps = reps

	FRAGS = fragment[]
	sizehint!(FRAGS, 1)
	
	if length(x0) != 0
		println("Initial conditions found, using first CSA fragment...")
		#@show x0
		α = 2
		xeff = x0[1:xsize]
		frag = fragment(xeff[fcl+1:end], xeff[1:fcl], 1, n_qubit, spin_orb)
		push!(FRAGS, frag)
		curr_obt, curr_tbt = fragment_to_tbt(frag)
	else
		α = 1
		curr_tbt = 0 .* tbt
		curr_obt = 0 .* obt
	end

	ini_cost = SD_cost(0, 0, obt, tbt)
	cost = SD_cost(curr_obt, curr_tbt, obt, tbt)
	
	FCost = SharedArray(zeros(tot_reps))
	X_ARR = SharedArray(zeros(tot_reps, xsize))
	println("Starting singles-doubles tensors cost is $ini_cost")
	while α <= α_max && cost > decomp_tol
		println("Starting parallel calculation for fragment number $α")
		t00 = time()
		@sync @distributed for idx in 1:tot_reps
			sol = greedy_step_SD_optimization(obt - curr_obt, tbt - curr_tbt, 1, frag_flavour=frag_flavour,
								 u_flavour=u_flavour, grad=grad, spin_orb=spin_orb, n = n)
			FCost[idx] = sol.minimum
			X_ARR[idx,:] .= sol.minimizer
		end
		
		ind_min = sortperm(FCost)[1]
		cost = FCost[ind_min]
		x0 = X_ARR[ind_min,:]
		frag = fragment(x0[fcl+1:end], x0[1:fcl], 1, n_qubit, spin_orb)
		tmp_obt, tmp_tbt = fragment_to_tbt(frag)
		curr_obt += tmp_obt
		curr_tbt += tmp_tbt
		push!(FRAGS, frag)
		println("Finished calculation for fragment $α after $(time()-t00) seconds, current cost is $cost")
		if verbose == true
			println("X vector for fragment $α is:")
			@show x0
		end
		
		if saving == true
			K0 = ones(Int64, α)
			if α > 1
				x_old = h5read(DATAFOLDER*f_name*".h5", "x0")
				x_old = append!(x_old, x0)
			else
				x_old = x0
			end
			overwrite_xK(f_name,x_old,K0)
		end

		α += 1
	end

	println("Found fragment, current cost is $cost, approximates initial cost by $(round(100*abs(ini_cost-cost)/ini_cost,digits=3))%")
	
	return FRAGS
end