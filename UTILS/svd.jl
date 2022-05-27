function tbt_svd(tbt :: Array; tol=1e-6, spin_orb=true, tiny=SVD_tiny, ret_op = true)
	println("Starting SVD routine")
	n = size(tbt)[1]
	N = n^2


	tbt_full = reshape(tbt, (N,N))
	tbt_res = Symmetric(reshape(tbt, (N,N)))
	if sum(abs.(tbt_full - tbt_res)) > tiny
		println("Non-symmetric two-body tensor as input for SVD routine, calculations might have errors...")
		tbt_res = tbt_full
	end

	println("Diagonalizing two-body tensor")
	@time Λ,U = eigen(tbt_res)
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res
	
	L_mats = Array{Complex{Float64},2}[]
	sizehint!(L_mats, N)
    
	FLAGS = zeros(Int64,N)

    for i in 1:N
    	if abs(Λ[i]) < tol
    		println("Truncating SVD for coefficients with magnitude smaller or equal to $(abs(Λ[i])), using $(i-1) fragments")
    		break
    	end
        cur_l = Symmetric(reshape(U[:, i], (n,n)))
        sym_dif = sum(abs.(cur_l - reshape(U[:, i], (n,n))))
        if sym_dif > tiny
        	cur_l = reshape(U[:, i], (n,n))
        	FLAGS[i] = 1
        end
        push!(L_mats, cur_l)
    end

    num_ops = length(L_mats)

    TBTS = SharedArray(zeros(Complex{Float64}, num_ops, n, n, n, n))
    CARTAN_TBTS = SharedArray(zeros(Complex{Float64}, num_ops, n, n, n, n))
    #U_MATS = SharedArray(zeros(num_ops, n, n, n, n))
    @sync @distributed for i in 1:num_ops
    	if FLAGS[i] == 0
	    	ωl, Ul = eigen(L_mats[i])

		    tbt_svd_CSA = zeros(typeof(ωl[1]),n,n,n,n)
		    for i1 in 1:n
		    	tbt_svd_CSA[i1,i1,i1,i1] = ωl[i1]^2
		    end

		    for i1 in 1:n
		    	for i2 in i1+1:n
		    		tbt_svd_CSA[i1,i1,i2,i2] = ωl[i1]*ωl[i2]
		    		tbt_svd_CSA[i2,i2,i1,i1] = ωl[i1]*ωl[i2]
		    	end
		    end

		    tbt_svd_CSA .*= Λ[i]
		    #println("Rotating tbt")
		    tbt_svd = unitary_cartan_rotation_from_matrix(Ul, tbt_svd_CSA)
		    #u_params = orb_rot_mat_to_params(Ul, n)
		    TBTS[i,:,:,:,:] = tbt_svd
		    CARTAN_TBTS[i,:,:,:,:] = tbt_svd_CSA
		else
			if sum(abs.(L_mats[i] + L_mats[i]')) > tiny
				error("SVD operator $i if neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!")
			end
			cur_l = Hermitian(1im * L_mats[i])
			# final operator is Lop = Λ[i] * cur_l * cur_l
			# cur_l = ±i*cur_l -> Lop = -Lop
			ωl, Ul = eigen(cur_l)
			
		    tbt_svd_CSA = zeros(typeof(ωl[1]),n,n,n,n)
		    for i1 in 1:n
		    	tbt_svd_CSA[i1,i1,i1,i1] = -ωl[i1]^2
		    end

		    for i1 in 1:n
		    	for i2 in i1+1:n
		    		tbt_svd_CSA[i1,i1,i2,i2] = -ωl[i1]*ωl[i2]
		    		tbt_svd_CSA[i2,i2,i1,i1] = -ωl[i1]*ωl[i2]
		    	end
		    end
		    
		    tbt_svd_CSA .*= Λ[i]
		    #println("Rotating tbt")
		    tbt_svd = unitary_cartan_rotation_from_matrix(Ul, tbt_svd_CSA)
		    #u_params = orb_rot_mat_to_params(Ul, n)
		    TBTS[i,:,:,:,:] = tbt_svd
		    CARTAN_TBTS[i,:,:,:,:] = tbt_svd_CSA
		end
	end

	println("Finished SVD routine")
	
	# = sanity routine, makes sure fragments recover initial tensor
	tbt_tot = TBTS[1,:,:,:,:]
	for i in 2:num_ops
		tbt_tot += TBTS[i,:,:,:,:]
	end
	sanity_sum = sum(abs.(tbt - tbt_tot))

	if sanity_sum > tiny
		println("Error, SVD routine did not decompose operator to correct accuracy, difference is $sanity_sum")
	else
		println("SVD sanity checked")
	end
	# =#
	if ret_op == false
		return CARTAN_TBTS, TBTS
	else
	    L_ops = []
	    for i in 1:length(L_mats)
	    	op_1d = obt_to_ferm(L_mats[i], spin_orb)
	    	push!(L_ops, Λ[i] * op_1d * op_1d)
	    end
		return CARTAN_TBTS, TBTS, sum(L_ops)
	end
end

function SVD_to_CSA(λ, ω, U; debug=false)
	#transform λ value, ω vec and U rotation coming from SVD into CSA fragment parameters
	#always returns MF_real rotation, rounds complex rotations to real
	n = length(ω)

	fcl = frag_coeff_length(n, CSA())
	coeffs = zeros(fcl)
	u_num = unitary_parameter_number(n, MF_real())

	idx = 1
	for i in 1:n
		for j in i:n
			coeffs[idx] = ω[i]*ω[j]
			idx += 1
		end
	end
	coeffs .*= λ

	U_log = real.(log(U))
	u_params = zeros(u_num)
	idx = 1
	for i in 1:n
		for j in i+1:n
			u_params[idx] = U_log[i,j]
			idx += 1
		end
	end
	
	if debug == true
		tbt_svd_CSA = zeros(typeof(ω[1]),n,n,n,n)
		for i1 in 1:n
		   	tbt_svd_CSA[i1,i1,i1,i1] = ω[i1]^2
	    end

	    for i1 in 1:n
	    	for i2 in i1+1:n
	    		tbt_svd_CSA[i1,i1,i2,i2] = ω[i1]*ω[i2]
	    		tbt_svd_CSA[i2,i2,i1,i1] = ω[i1]*ω[i2]
	    	end
	    end
		    
	    tbt_svd_CSA .*= λ
	    tbt_svd = unitary_cartan_rotation_from_matrix(U, tbt_svd_CSA)

	    frag = fragment(u_params, coeffs, 1, n, true)
	    tbt_frag = fragment_to_tbt(frag, frag_flavour=CSA(), u_flavour=MF_real())
	    @show sum(abs.(tbt_svd - tbt_frag))
	end

	return coeffs, u_params
end


function tbt_svd_1st(tbt :: Array; debug=false, tiny=SVD_tiny)
	#returns the largest SVD component from tbt in CSA fragment coefficients
	#println("Starting tbt_svd_1st routine!")
	n = size(tbt)[1]
	N = n^2

	tbt_res = Symmetric(reshape(tbt, (N,N)))

	#println("Diagonalizing")
	Λ,U = eigen(tbt_res)
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res

	full_l = reshape(U[:, 1], (n,n))
    cur_l = Symmetric(full_l)

    sym_dif = sum(abs.(cur_l - full_l))
    if sym_dif > tiny
      	if sum(abs.(full_l + full_l')) > tiny
			error("SVD operator is neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!")
		end
      	
       	cur_l = Hermitian(1im * full_l)
       	Λ[1] *= -1
    end

    ωl, Ul = eigen(cur_l)
    
    return SVD_to_CSA(Λ[1], ωl, Ul)
end

function svd_optimized(tbt :: Array; tol=1e-6, spin_orb=true, tiny=SVD_tiny)
	#do svd and calculate L1 norms in naive way (i.e. separated terms)
	println("Starting SVD routine")
	n = size(tbt)[1]
	N = n^2


	tbt_full = reshape(tbt, (N,N))
	tbt_res = Symmetric(reshape(tbt, (N,N)))
	if sum(abs.(tbt_full - tbt_res)) > tiny
		println("Non-symmetric two-body tensor as input for SVD routine, calculations might have errors...")
		tbt_res = tbt_full
	end

	println("Diagonalizing two-body tensor")
	@time Λ,U = eigen(tbt_res)
	#@show (maximum(Λ) - minimum(Λ))/2
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res
	
	global num_ops = N
    for i in 1:N
    	if abs(Λ[i]) < tol
    		println("Truncating SVD for coefficients with magnitude smaller or equal to $(abs(Λ[i])), using $(i-1) fragments")
    		global num_ops = i-1
    		break
    	end
	end
	Λ = Λ[1:num_ops]
	U = U[:,1:num_ops]

	SR_L1_RANGES = SharedArray(zeros(num_ops,2))
	NR_L1 = SharedArray(zeros(num_ops))
	ωs_arr = SharedArray(zeros(num_ops, n))
	@sync @distributed for i in 1:num_ops
        full_l = reshape(U[:, i], (n,n))
        cur_l = Symmetric(full_l)
        sym_dif = sum(abs.(cur_l - full_l))
        if sym_dif > tiny
        	if sum(abs.(full_l + full_l')) > tiny
				error("SVD operator $i if neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!")
			end
        	
        	cur_l = Hermitian(1im * full_l)
        	Λ[i] *= -1
        end
    
		ωs_arr[i,:], _ = eigen(cur_l)

	    ω_pos = (ωs_arr[i,:] + abs.(ωs_arr[i,:]))/2
	    ω_neg = ωs_arr[i,:] - ω_pos
	    max_norm = Λ[i] * (maximum([sum(abs.(ω_pos)), sum(abs.(ω_neg))])^2)
	    if max_norm > 0
	    	SR_L1_RANGES[i,2] = max_norm
	    else
	    	SR_L1_RANGES[i,1] = max_norm
	    end

	    val_red = 0.0
	    NR_L1_red = zeros(num_ops)
	    curr_val = 0.0
	    for k1 in 1:n
	    	for k2 in k1+1:n
	    		curr_val += 2*abs(ωs_arr[i,k1]*ωs_arr[i,k2])
	    		val_red += 2*abs(ωs_arr[i,k1]*ωs_arr[i,k2])
	    	end
	    end
	    for k1 in 1:n
	    	curr_val += abs(ωs_arr[i,k1]^2)
	    end
	    NR_L1[i] = abs(Λ[i]) * curr_val
	    NR_L1_red[i] = abs(Λ[i]) * val_red
			
	end

	println("Finished SVD routine")
	if spin_orb == false
		NR_L1 *= 4
		NR_L1_red *= 4
		SR_L1_RANGES *= 4
	end

	λV = sum(NR_L1)
	λV_red = sum(NR_L1_red)

	@show λV, λV_red

	return NR_L1, SR_L1_RANGES
end

function svd_optimized_df(tbt :: Tuple; tol=1e-6, tiny=SVD_tiny)
	#do svd and calculate L1 norms following double rank factorization
	#uses spacial orbitals (i.e. spin_orb = false)
	println("Starting SVD routine")
	n = size(tbt[1])[1]
	N = n^2


	tbt_full = reshape(tbt[2], (N,N))
	tbt_res = Symmetric(reshape(tbt[2], (N,N)))
	if sum(abs.(tbt_full - tbt_res)) > tiny
		println("Non-symmetric two-body tensor as input for SVD routine, calculations might have errors...")
		tbt_res = tbt_full
	end

	println("Diagonalizing two-body tensor")
	@time Λ,U = eigen(tbt_res)
	#@show (maximum(Λ) - minimum(Λ))/2
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res
	
	global num_ops = N
    for i in 1:N
    	if abs(Λ[i]) < tol
    		println("Truncating SVD for coefficients with magnitude smaller or equal to $(abs(Λ[i])), using $(i-1) fragments")
    		global num_ops = i-1
    		break
    	end
	end
	Λ = Λ[1:num_ops]
	U = U[:,1:num_ops]

	ωs_arr = SharedArray(zeros(num_ops, n))
	@sync @distributed for i in 1:num_ops
        full_l = reshape(U[:, i], (n,n))
        cur_l = Symmetric(full_l)
        sym_dif = sum(abs.(cur_l - full_l))
        if sym_dif > tiny
        	if sum(abs.(full_l + full_l')) > tiny
				error("SVD operator $i is neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!")
			end
        	
        	cur_l = Hermitian(1im * full_l)
        	Λ[i] *= -1
        end
    
		ωs_arr[i,:], _ = eigen(cur_l)
		ωs_arr[i,:] *= sqrt(2*abs(Λ[i]))
	end

	obt_tilde = tbt[1] + 2*sum([tbt[2][:,:,r,r] for r in 1:n])
	obt_D, _ = eigen(obt_tilde)
    λT = sum(abs.(obt_D))
    
    λDF = 0.0
    for i in 1:num_ops
    	λDF += sum(abs.(ωs_arr[i,:]))^2
    end
    λDF /= 4
    

    @show λT, λDF
    @show λT + λDF
    
    return λT + λDF
end

function svd_optimized_so(tbt_so; tol=1e-6, tiny=SVD_tiny)
	#do svd and calculate L1 norms following double rank factorization
	#uses spacial orbitals (i.e. spin_orb = false)
	println("Starting SVD routine")
	n = size(tbt_so)[1]
	N = n^2


	tbt_full = reshape(tbt_so, (N,N))
	tbt_res = Symmetric(tbt_full)
	if sum(abs.(tbt_full - tbt_res)) > tiny
		println("Non-symmetric two-body tensor as input for SVD routine, calculations might have errors...")
		tbt_res = tbt_full
	end

	println("Diagonalizing two-body tensor")
	@time Λ,U = eigen(tbt_res)
	#@show (maximum(Λ) - minimum(Λ))/2
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res
	
	global num_ops = N
    for i in 1:N
    	if abs(Λ[i]) < tol
    		println("Truncating SVD for coefficients with magnitude smaller or equal to $(abs(Λ[i])), using $(i-1) fragments")
    		global num_ops = i-1
    		break
    	end
	end
	Λ = Λ[1:num_ops]
	U = U[:,1:num_ops]

	ωs_arr = SharedArray(zeros(num_ops, n))
	@sync @distributed for i in 1:num_ops
        full_l = reshape(U[:, i], (n,n))
        cur_l = Symmetric(full_l)
        sym_dif = sum(abs.(cur_l - full_l))
        if sym_dif > tiny
        	if sum(abs.(full_l + full_l')) > tiny
				error("SVD operator $i is neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!")
			end
        	
        	cur_l = Hermitian(1im * full_l)
        	Λ[i] *= -1
        end
    	
		ωs_arr[i,:], Ul = eigen(cur_l)
		ωs_arr[i,:] *= sqrt(abs(Λ[i]))
	end

	obt = sum([tbt_so[:,:,r,r] for r in 1:n])
	obt_D, _ = eigen(obt)
    λT = sum(abs.(obt_D))
    
    λDF = 0.0
    λDF_red = 0.0
    for i in 1:num_ops
    	λDF += sum(abs.(ωs_arr[i,:]))^2
    	λDF_red -= sum(abs2.(ωs_arr[i,:]))
    end
    λDF /= 4
    λDF_red /= 4
    λDF_red += λDF

    @show λT
    @show λDF, λDF_red
    @show λT + λDF
    @show λT + λDF_red

    return λT + λDF
end