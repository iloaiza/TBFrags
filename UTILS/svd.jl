function tbt_svd(tbt :: Array; tol=1e-6, spin_orb=false)
	println("Starting SVD routine")
	n = size(tbt)[1]
	N = n^2

	tbt_res = Symmetric(reshape(tbt, (N,N)))

	println("Diagonalizing two-body tensor")
	@time Λ,U = eigen(tbt_res)
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res
	
	L_mats = Array{Complex{Float64},2}[]
	sizehint!(L_mats, N)
    
    for i in 1:N
    	if abs(Λ[i]) < tol
    		println("Breaking for $(Λ[i])")
    		break
    	end
        cur_l = Symmetric(reshape(U[:, i], (n,n)))
        push!(L_mats, cur_l)
    end

    num_ops = length(L_mats)
    @show num_ops

    L_ops = []
    for i in 1:length(L_mats)
    	op_1d = obt_to_ferm(L_mats[i], spin_orb)
    	push!(L_ops, Λ[i] * op_1d * op_1d)
    end

    TBTS = SharedArray(zeros(Complex{Float64}, num_ops, n, n, n, n))
    CARTAN_TBTS = SharedArray(zeros(Complex{Float64}, num_ops, n, n, n, n))
    #U_MATS = SharedArray(zeros(num_ops, n, n, n, n))
    @sync @distributed for i in 1:num_ops
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
	end

	println("Finished SVD routine")
	return CARTAN_TBTS, TBTS
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

