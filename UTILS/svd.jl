function tbt_svd(tbt :: Array; tol=1e-6, spin_orb=true, tiny=1e-8)
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
    L_ops = []
    for i in 1:length(L_mats)
    	op_1d = obt_to_ferm(L_mats[i], spin_orb)
    	push!(L_ops, Λ[i] * op_1d * op_1d)
    end

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
	println("SVD sanity:")
	tbt_tot = TBTS[1,:,:,:,:]
	for i in 2:num_ops
		tbt_tot += TBTS[i,:,:,:,:]
	end
	sanity_sum = sum(abs.(tbt - tbt_tot))

	if sanity_sum > tiny
		println("Error, SVD routine did not decompose operator to correct accuracy, difference is $sanity_sum")
	end

	return CARTAN_TBTS, TBTS, sum(L_ops)
end

function tbt_svd_1st(tbt :: Array; spin_orb=false, debug=false, return_CSA=false)
	#returns the largest SVD component from tbt
	n = size(tbt)[1]
	N = n^2

	tbt_res = Symmetric(reshape(tbt, (N,N)))

	println("Diagonalizing")
	@time Λ,U = eigen(tbt_res)
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res
    L_mat = Symmetric(reshape(U[:, 1], (n,n)))

    ωl, Ul = eigen(L_mat)
    tbt_svd_greedy_CSA = zeros(n,n,n,n)
    for i in 1:n
    	tbt_svd_greedy_CSA[i,i,i,i] = ωl[i]^2
    end

    for i in 1:n
    	for j in i+1:n
    		tbt_svd_greedy_CSA[i,i,j,j] = ωl[i]*ωl[j]
    		tbt_svd_greedy_CSA[j,j,i,i] = ωl[i]*ωl[j]
    	end
    end

    tbt_svd_greedy_CSA .*= Λ[1]
    tbt_svd_greedy = unitary_cartan_rotation_from_matrix(Ul, tbt_svd_greedy_CSA)

    if debug == true
    	CSA_op = tbt_to_ferm(tbt_svd_greedy, spin_orb)
    	op_1d = obt_to_ferm(L_mat, spin_orb)
    	op_2d = Λ[i] * op_1d * op_1d
    	@show of_simplify(op_2d - CSA_op)
    end
    
    if return_CSA == false
    	return tbt_svd_greedy
    else
    	return tbt_svd_greedy, tbt_svd_greedy_CSA, orb_rot_mat_to_params(Ul)
    end
end

