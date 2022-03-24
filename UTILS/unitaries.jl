#FUNCTIONS FOR BUILDING UNITARIES AND ROTATING TWO-BODY TENSORS USING UNITARIES

function get_anti_symmetric(n, N = Int(n*(n-1)/2))
	# Construct list of anti-symmetric matrices kappa_pq based on n*(n-1)/2 
	R = zeros(N,n,n)
	idx = 1
	for p in 1:n
		for q in p+1:n
			R[idx,p,q] = 1
			R[idx,q,p] = -1
			idx += 1
		end
	end

	return R
end

function construct_anti_symmetric(n, params, N=length(params))
	#Constrcut the nxn anti-symmetric matrix based on the sum of basis with params as coefficients
	real_anti = get_anti_symmetric(n, N)
	anti_symm = zeros(n,n)
	for idx in 1:N
		anti_symm += params[idx] * real_anti[idx,:,:]
	end

	return anti_symm
end

function MF_real_generator(n, params, N=length(params))
	anti_symm = construct_anti_symmetric(n, params, N)

	return anti_symm
end

function MF_real_unitary(n, params, N = length(params))
	#Construct nxn unitary by exponentiating anti-symmetric matrix
	anti_symm = MF_real_generator(n, params, N)

	return exp(anti_symm)
end

function get_imag_symmetric(n, N = Int(n*(n-1)/2))
    #Construct list of i*symmetric matrices kappa_pq based on n*(n-1)/2 
    I = zeros(Complex{Float64},N,n,n)
    idx = 1
    for p in 1:n
        for q in p+1:n
            I[idx, p, q] = 1im
            I[idx, q, p] = 1im
            idx += 1
        end
    end

    return I
end

function MF_generator(n, params, N=length(params))
	#The parameters are n(n-1) terms that determines the lower diagonals of e^{i symm + anti-symmetric}
    
    anti_herm = zeros(Complex{Float64},n,n)
    full_anti = zeros(Complex{Float64},N,n,n)
    Nsmall = Int(N/2)
    full_anti[1:Nsmall,:,:] = get_imag_symmetric(n)
    full_anti[Nsmall+1:end,:,:] = get_anti_symmetric(n)
    
    for idx in 1:N
        anti_herm += params[idx] * full_anti[idx,:,:]
    end

    return anti_herm
end

function MF_unitary(n, params, N = length(params))
    #The parameters are n(n-1) terms that determines the lower diagonals of e^{i symm + anti-symmetric}
    #=
    anti_herm = zeros(Complex{Float64},n,n)
    full_anti = zeros(Complex{Float64},N,n,n)
    Nsmall = Int(N/2)
    full_anti[1:Nsmall,:,:] = get_imag_symmetric(n)
    full_anti[Nsmall+1:end,:,:] = get_anti_symmetric(n)
    
    for idx in 1:N
        anti_herm += params[idx] * full_anti[idx,:,:]
    end
    # =#

    anti_herm = MF_generator(n, params, N)

    return exp(anti_herm)
end

function unitary_rotation_matrix(u_params, n; u_flavour=META.uf)
	# generate unitary matrix Umat from flavour and parameters
	if typeof(u_flavour) == MF_real
		Umat = MF_real_unitary(n, u_params)
	elseif typeof(u_flavour) == MF
		Umat = MF_unitary(n, u_params)
	else
		error("Trying to obtain unitary rotation matrix with flavour $(u_flavour), not implemented!")
	end

	return Umat
end

function unitary_generator(u_params, n; u_flavour=META.uf)
	if typeof(u_flavour) == MF_real
		Ugen = MF_real_generator(n, u_params)
	elseif typeof(u_flavour) == MF
		Ugen = MF_generator(n, u_params)
	else
		error("Trying to obtain unitary rotation matrix with flavour $(u_flavour), not implemented!")
	end

	return Ugen
end

function unitary_rotation(u_params, tbt, n=length(tbt[:,1,1,1]), u_flavour=META.uf)
	# warning: only works for tbt consisting of Cartans (i.e. tbt[i,j,k,l] ∝ δij*δkl)
	# return rotation of tbt with unitary given by u_params, using unitary flavour u_flavour

	# generate unitary matrix Umat from flavour and parameters
	if typeof(u_flavour) == MF_real
		Umat = MF_real_unitary(n, u_params)
	elseif typeof(u_flavour) == MF
		Umat = MF_unitary(n, u_params)
	else
		error("Trying to perform unitary rotation with flavour $(u_flavour), not implemented!")
	end

	# generate rotated two-body tensor
	if typeof(u_flavour) == MF_real
		rotated_tbt = zeros(typeof(tbt[1,1,1,1]),n,n,n,n)
		@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * Umat[b,l] * Umat[c,m] * Umat[d,m] * tbt[l,l,m,m]
	elseif typeof(u_flavour) == MF
		rotated_tbt = zeros(Complex{Float64},n,n,n,n)
		@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * conj(Umat[b,l]) * Umat[c,m] * conj(Umat[d,m]) * tbt[l,l,m,m]
	end

	return rotated_tbt
end

function unitary_SD_rotation(u_params, obt, tbt, n=length(tbt[:,1,1,1]), u_flavour=META.uf)
	# warning: only works for tbt consisting of Cartans (i.e. tbt[i,j,k,l] ∝ δij*δkl)
	# return rotation of tbt with unitary given by u_params, using unitary flavour u_flavour

	# generate unitary matrix Umat from flavour and parameters
	if typeof(u_flavour) == MF_real
		Umat = MF_real_unitary(n, u_params)
	elseif typeof(u_flavour) == MF
		Umat = MF_unitary(n, u_params)
	else
		error("Trying to perform unitary rotation with flavour $(u_flavour), not implemented!")
	end

	# generate rotated two-body tensor
	if typeof(u_flavour) == MF_real
		rotated_tbt = zeros(typeof(tbt[1,1,1,1]),n,n,n,n)
		@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * Umat[b,l] * Umat[c,m] * Umat[d,m] * tbt[l,l,m,m]
		rotated_obt = zeros(typeof(obt[1,1]),n,n)
		@einsum rotated_obt[a,b] = Umat[a,l] * Umat[b,l] * obt[l,l]
	elseif typeof(u_flavour) == MF
		rotated_tbt = zeros(Complex{Float64},n,n,n,n)
		@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * conj(Umat[b,l]) * Umat[c,m] * conj(Umat[d,m]) * tbt[l,l,m,m]
		rotated_obt = zeros(Complex{Float64},n,n)
		@einsum rotated_obt[a,b] = Umat[a,l] * conj(Umat[b,l]) * obt[l,l]
	end

	return rotated_obt, rotated_tbt
end

function generalized_unitary_rotation(u_params, tbt, n=length(tbt[:,1,1,1]), u_flavour=META.uf)
	# as unitary rotation, works for non-Cartan two-body-tensors too
	# return rotation of tbt with unitary given by u_params, using unitary flavour u_flavour
	
	# generate unitary matrix Umat from flavour and parameters
	if typeof(u_flavour) == MF_real
		Umat = MF_real_unitary(n, u_params)
	elseif typeof(u_flavour) == MF
		Umat = MF_unitary(n, u_params)
	else
		error("Trying to perform unitary rotation with flavour $(u_flavour), not implemented!")
	end

	# generate rotated two-body tensor
	if typeof(u_flavour) == MF_real
		rotated_tbt = zeros(typeof(tbt[1,1,1,1]),n,n,n,n)
		@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * Umat[b,k] * Umat[c,m] * Umat[d,p] * tbt[l,k,m,p]
	elseif typeof(u_flavour) == MF
		rotated_tbt = zeros(Complex{Float64},n,n,n,n)
		@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * conj(Umat[b,k]) * Umat[c,m] * conj(Umat[d,p]) * tbt[l,k,m,p]
	end

	return rotated_tbt
end

function unitary_parameter_number(n, flavour = META.uf)
	if typeof(flavour) == MF_real
		return Int(n*(n-1)/2)
	elseif typeof(flavour) == MF
		return Int(n*(n-1))
	else
		error("Trying to obtain number of parameters for unitary flavour $flavour, not defined!")
	end
end