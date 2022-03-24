function ccd_cost(u_params, tbt; n=length(tbt[:,1,1,1]), u_flavour=META.uf)
	tbt_rot = generalized_unitary_rotation(u_params, tbt, n, u_flavour)

	return -maximum(abs.(tbt_rot))
end

function greedy_ccd_mf(tbt, α_max, u_flavour=META.uf)
	n = size(tbt)[1]
	u_length = unitary_parameter_number(n, u_flavour)

	FRAGS = fragment[]

	curr_tbt = 0 .* tbt
	for i in 1:α_max
		function cost(x)
			return ccd_cost(x, tbt-curr_tbt, n=n, u_flavour=u_flavour)
		end
		x0 = 2π*rand(u_length)
		sol = optimize(cost, x0, BFGS())

		max_val = -sol.minimum
		println("Absolute amplitude without rotation is $(maximum(abs.(tbt - curr_tbt))), amplitude with rotation is $max_val")

		rot_tbt = generalized_unitary_rotation(sol.minimizer, tbt-curr_tbt, n, u_flavour)
		Nidx = argmax(abs.(rot_tbt))
		cc_sign = sign(rot_tbt[Nidx])

		cc_tbt = zeros(n,n,n,n)
		cc_tbt[Nidx] = 1
		cc_tbt = generalized_unitary_rotation(-sol.minimizer, cc_tbt, n, u_flavour)
		curr_tbt += cc_tbt

		cc_frag = fragment(-sol.minimizer, [1im*cc_sign*max_val,Nidx[1],Nidx[2],Nidx[3],Nidx[4]], 1, n, true)

		push!(FRAGS, cc_frag)
	end

	return FRAGS
end

