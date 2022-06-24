# FUNCTIONS FOR CALCULATING COST OF TBT AND FRAGMENTS W/R TO TARGET
import Base.+
import Base.-

+(A::Tuple, B::Tuple) = (A[1]+B[1], A[2]+B[2])
-(A::Tuple, B::Tuple) = (A[1]-B[1], A[2]-B[2])

function cartan_obt_l1_cost(obt :: Array, spin_orb=true)
	if length(size(obt)) == 1
		#obt is just array of diagonal elements
		l1_cost = sum(abs.(obt))
	else	
		n=size(obt)[1]
		l1_cost = sum(abs.(Diagonal(obt)))
	end

	if spin_orb
		return l1_cost
	else
		return 2*l1_cost
	end
end

function cartan_so_tbt_l1_cost(tbt :: Array)
	#find tbt cost using np*nq -> (1-2np)(1-2nq) unitarization
	#requires spin-orbitals
	n=size(tbt)[1]

	global l1_cost = 0.0
	for i in 1:n
		for j in i+1:n
			global l1_cost += abs(tbt[i,i,j,j])
			global l1_cost += abs(tbt[j,j,i,i])
		end
	end

	l1_cost
end


function cartan_tbt_l1_cost(tbt :: Array, spin_orb=true)
	n=size(tbt)[1]

	global l1_cost = 0.0
	global diag_cost = 0.0
	for i in 1:n
		global diag_cost += abs(tbt[i,i,i,i])
		for j in i+1:n
			global l1_cost += abs(tbt[i,i,j,j])
			global l1_cost += abs(tbt[j,j,i,i])
		end
	end

	if spin_orb
		return l1_cost
	else
		return 4*l1_cost - 2*diag_cost
	end
end

function cartan_tbt_l1_cost(tbt :: Tuple, spin_orb=true)
	tbt_new = (tbt[1], tbt[2])
	n = size(tbt[1])[1]

	if spin_orb == true
		s_factor = 1
	else
		s_factor = 2
	end

	for i in 1:n
		tbt_new[1][i,i] += s_factor*tbt[2][i,i,i,i]
		tbt_new[2][i,i,i,i] = 0
	end

	return cartan_obt_l1_cost(tbt_new[1], spin_orb) + cartan_tbt_l1_cost(tbt_new[2], spin_orb)
end

function cartan_obt_l2_cost(obt :: Array, spin_orb=true)
	n=size(obt)[1]
	l2_cost = sum(abs2.(Diagonal(obt)))

	if spin_orb
		return l2_cost
	else
		return 2*l2_cost
	end
end

function cartan_tbt_l2_cost(tbt :: Array, spin_orb=true)
	n=size(tbt)[1]

	global l2_cost = 0.0
	for i in 1:n
		for j in 1:n
			global l2_cost += abs(tbt[i,i,j,j])
		end
	end

	if spin_orb
		return l2_cost
	else
		return 4*l2_cost
	end
end

function cartan_tbt_l2_cost(tbt :: Tuple, spin_orb=true)
	tbt_new = (tbt[1], tbt[2])
	n = size(tbt[1])[1]

	if spin_orb == true
		s_factor = 1
	else
		s_factor = 2
	end

	for i in 1:n
		tbt_new[1][i,i] += s_factor*tbt[2][i,i,i,i]
		tbt_new[2][i,i,i,i] = 0
	end

	return cartan_obt_l2_cost(tbt_new[1], spin_orb) + cartan_tbt_l2_cost(tbt_new[2], spin_orb)
end

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

function tbt_cost(tbt, target :: Tuple)
	if tbt == 0
		return SD_cost(0, 0, target[1], target[2])
	else
		error("Trying to obtain tbt cost where target is a touple but tbt isn't")
	end
end

function tbt_cost(tbt :: Tuple, target :: Tuple)
	return SD_cost(tbt[1], tbt[2], target[1], target[2])
end

function tbt_cost(tbt)
	return tbt_cost(0, tbt)
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
	tbt = 0 .* target
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
	tbt = 0 .* target
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
	tbt = 0 .* target
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