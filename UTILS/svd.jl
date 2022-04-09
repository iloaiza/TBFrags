function tbt_svd(tbt :: Array; tol=1e-6, spin_orb=false, imag_tol=1e-12)
	println("Starting SVD routine")
	n = size(tbt)[1]
	N = n^2

	tbt_res = reshape(tbt, (N,N))

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
        cur_l = sqrt(complex(Λ[i])) * reshape(U[:, i], (n,n))
        push!(L_mats, cur_l)
    end

    num_ops = length(L_mats)
    @show num_ops

    L_ops = []
    for i in 1:length(L_mats)
    	op_1d = obt_to_ferm(L_mats[i], spin_orb)
    	push!(L_ops, op_1d * op_1d)
    end

    TBTS = SharedArray(zeros(Complex{Float64}, num_ops, n, n, n, n))
    CARTAN_TBTS = SharedArray(zeros(Complex{Float64}, num_ops, n, n, n, n))
    #U_MATS = SharedArray(zeros(num_ops, n, n, n, n))
    @sync @distributed for i in 1:num_ops
    	#println("Diagonalizing $i")
    	#@show typeof(L_mats[i][1])
        #@show sum(abs.(imag.(L_mats[i]))), sum(abs.(real.(L_mats[i])))
    	ωl, Ul = eigen(L_mats[i])
	    tbt_svd_greedy_CSA = zeros(Complex{Float64},n,n,n,n)
	    for i in 1:n
	    	tbt_svd_greedy_CSA[i,i,i,i] = ωl[i]^2
	    end

	    for i in 1:n
	    	for j in i+1:n
	    		tbt_svd_greedy_CSA[i,i,j,j] = ωl[i]*ωl[j]
	    		tbt_svd_greedy_CSA[j,j,i,i] = ωl[i]*ωl[j]
	    	end
	    end

	    if sum(abs.(imag.(tbt_svd_greedy_CSA))) < imag_tol
	    	tbt_svd_greedy_CSA = real.(tbt_svd_greedy_CSA)
	    	#println("All real for $i")
	    end

	    if sum(abs.(imag.(Ul))) < imag_tol
	    	Ul = real.(Ul)
	    	#println("All real for unitary $i")
	    end

	    #println("Rotating tbt")
	    tbt_svd_greedy = unitary_cartan_rotation_from_matrix(Ul, tbt_svd_greedy_CSA)
	    #u_params = orb_rot_mat_to_params(Ul, n)
	    TBTS[i,:,:,:,:] = tbt_svd_greedy
	    CARTAN_TBTS[i,:,:,:,:] = tbt_svd_greedy_CSA
	end

	println("Finished SVD routine")
	return CARTAN_TBTS, TBTS
end

function tbt_svd_1st(tbt :: Array; spin_orb=false, debug=false, return_CSA=false)
	#returns the largest SVD component from tbt
	n = size(tbt)[1]
	N = n^2

	tbt_res = reshape(tbt, (N,N))

	println("Diagonalizing")
	@time Λ,U = eigen(tbt_res)
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res
    L_mat = sqrt(complex(Λ[1])) * reshape(U[:, 1], (n,n))

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

    tbt_svd_greedy = unitary_cartan_rotation_from_matrix(Ul, tbt_svd_greedy_CSA)

    if debug == true
    	CSA_op = tbt_to_ferm(tbt_svd_greedy, spin_orb)
    	op_1d = obt_to_ferm(L_mat, spin_orb)
    	op_2d = op_1d*op_1d
    	@show of_simplify(op_2d - CSA_op)
    end
    
    if return_CSA == false
    	return tbt_svd_greedy
    else
    	return tbt_svd_greedy, tbt_svd_greedy_CSA, orb_rot_mat_to_params(Ul)
    end
end

