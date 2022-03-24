# FUNCTIONS FOR CALCULATING COST OF TBT AND FRAGMENTS W/R TO TARGET
function SD_cost(obt, tbt, ob_target, tb_target)
	if tbt == 0
		tb_diff = tb_target
	elseif tb_target == 0
		tb_diff = tbt
	else
		tb_diff = tbt - tb_target
	end

	if obt == 0
		ob_diff = ob_target
	elseif ob_target == 0
		ob_diff = obt
	else
		ob_diff = obt - ob_target
	end

	return sum(abs2.(ob_diff)) + sum(abs2.(tb_diff)) #L2 cost, works better
	#return sum(abs.(ob_diff)) + sum(abs.(tb_diff)) #L1 cost
end
		
function tbt_cost(tbt, target)
	if tbt == 0
		diff = target
	elseif target == 0
		diff = tbt
	else
		diff = tbt - target
	end

	return sum(abs2.(diff)) #L2 norm
	#return sum(abs2.(diff)) #L1 norm
end

function fragment_cost(frag, target; frag_flavour=META.ff, u_flavour=META.uf)
	return tbt_cost(fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour), target)
end

function parameter_to_frag(x, class, n; spin_orb = false, frag_flavour=META.ff, u_flavour=META.uf)
	fcl = frag_coeff_length(n, frag_flavour)
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	return frag = fragment(x[fcl+1:end], x[1:fcl], class, n_qubit, spin_orb)
end

function parameter_to_tbt(x, class, n; spin_orb = false, frag_flavour=META.ff, u_flavour=META.uf)
	frag = parameter_to_frag(x, class, n, spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)

	return fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
end

function SD_parameter_cost(x, ob_target, tb_target, class; n=size(ob_target)[1], spin_orb = false, frag_flavour=META.ff, u_flavour=META.uf)
	obt, tbt = parameter_to_tbt(x, class, n, spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)

	return SD_cost(obt, tbt, ob_target, tb_target)
end

function parameter_cost(x, target, class; n=length(target[:,1,1,1]), spin_orb = false, frag_flavour=META.ff, u_flavour=META.uf)
	tbt = parameter_to_tbt(x, class, n, spin_orb=spin_orb, frag_flavour=frag_flavour, u_flavour=u_flavour)

	return tbt_cost(tbt, target)
end

function full_rank_cost(x, target, class_arr; n=length(target[:,1,1,1]), spin_orb = false, frag_flavour=META.ff, u_flavour=META.uf)
	#x_arr = reshape(x, num_classes, :)
	tbt = zeros(n,n,n,n)
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	num_frags = length(class_arr)
	fcl = frag_coeff_length(n, frag_flavour)

	u_num = unitary_parameter_number(n, u_flavour)
	x_size = u_num + fcl
	x_ini = 1
	for i in 1:num_frags
		x0 = x[x_ini:x_ini+x_size-1]
		frag = fragment(x0[fcl+1:end], x0[1:fcl], class_arr[i], n_qubit, spin_orb)
		tbt += fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
		x_ini += x_size
	end

	return tbt_cost(tbt, target)
end

function relaxed_cost(x, target, class_arr, u_arr, x_prev; n=length(target[:,1,1,1]), spin_orb = false, frag_flavour=META.ff, u_flavour=META.uf)
	#uses unitaries from previous arrays, optimizes fully new fragment and previous fragments coefficients
	#x_prev: array of previous x, dimensions are [fcl-1, num_frags-1]
	#u_arr: array of previous unitaries, dimensions are [u_num, num_frags-1]
	tbt = zeros(n,n,n,n)
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	num_frags = length(class_arr)
	fcl = frag_coeff_length(n, frag_flavour)

	u_num = unitary_parameter_number(n, u_flavour)
	x_size = u_num + fcl
	for i in 1:num_frags-1
		x_eff = cat(x[i], x_prev[:, i], dims=1)
		frag = fragment(u_arr[:, i], x_eff, class_arr[i], n_qubit, spin_orb)
		tbt += fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end

	x0 = x[num_frags:end]
	frag = fragment(x0[fcl+1:end], x0[1:fcl], class_arr[end], n_qubit, spin_orb)
	tbt += fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
	
	return tbt_cost(tbt, target)
end

function relaxed_block_cost(x, target, class_arr, u_arr, x_prev; b_size = block_size, n=length(target[:,1,1,1]), spin_orb = false, frag_flavour=META.ff, u_flavour=META.uf)
	#uses unitaries from previous arrays, optimizes fully b_size new fragments and previous fragments coefficients
	#x_prev: array of previous x, dimensions are [fcl-1, num_frags-b_size]
	#u_arr: array of previous unitaries, dimensions are [u_num, num_frags-b_size]
	tbt = zeros(n,n,n,n)
	if spin_orb
		n_qubit = n
	else
		n_qubit = 2n
	end
	num_frags = length(class_arr)
	fcl = frag_coeff_length(n, frag_flavour)

	u_num = unitary_parameter_number(n, u_flavour)
	x_size = u_num + fcl
	for i in 1:num_frags-b_size
		x_eff = cat(x[i], x_prev[:, i], dims=1)
		frag = fragment(u_arr[:, i], x_eff, class_arr[i], n_qubit, spin_orb)
		tbt += fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end

	x_ini = num_frags-b_size+1
	for i in 1:b_size
		x0 = x[x_ini:x_ini+x_size-1]
		frag = fragment(x0[fcl+1:end], x0[1:fcl], class_arr[end], n_qubit, spin_orb)
		tbt += fragment_to_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
		x_ini += x_size
	end
	
	return tbt_cost(tbt, target)
end