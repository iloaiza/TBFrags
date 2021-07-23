# FUNCTIONS FOR CALCULATING COST OF TBT AND FRAGMENTS W/R TO TARGET

function tbt_cost(tbt, target)
	if tbt == 0
		diff = target
	elseif target == 0
		diff = tbt
	else
		diff = tbt - target
	end

	return sum(abs2.(diff))
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