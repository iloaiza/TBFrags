#FUNCTIONS FOR CALCULATING GRADIENTS FOR FASTER MINIMIZATIONS
function CSA_get_cartan_matrix(cn_vec, n)
    V = zeros(n,n)

    coeffs = cn_vec[2:end]
    idx = 1
    for i in 1:n
        for j in i:n
            V[i,j] = coeffs[idx]
            V[j,i] = coeffs[idx]
            idx += 1
        end
    end

    return V
end

function CGMFR_get_cartan_matrix(k, cn, n)
	V = zeros(n,n)

	if k==1
       	V[1,1] = -2
    elseif k==2
       	V[1,2] = -1
       	V[2,1] = -1
    elseif k==3
    	V[1,1] = -2
    	V[1,2] = 1
    	V[2,1] = 1
    elseif k==4
    	V[1,1] = -2
    	V[2,2] = -2
    	V[1,2] = 2
    	V[2,1] = 2
    elseif k==5
    	V[1,1] = -2
    	V[2,2] = -2
    	V[1,2] = 1
    	V[2,1] = 1
    elseif k==6
    	V[1,1] = -2
    	V[2,3] = -1
    	V[3,2] = -1
    	V[1,3] = 1
    	V[3,1] = 1
    elseif k==7
    	V[1,1] = -2
    	V[2,3] = -1
    	V[3,2] = -1
    	V[1,3] = 1
    	V[3,1] = 1
    	V[1,2] = 1
    	V[2,1] = 1
    elseif k==8
    	V[1,1] = -2
    	V[2,2] = -2
    	V[1,3] = 1
    	V[3,1] = 1
    	V[1,2] = 1
    	V[2,1] = 1
    elseif k==9
    	V[1,1] = -2
    	V[2,2] = -2
    	V[3,3] = -2
    	V[2,3] = 1
    	V[3,2] = 1
    	V[1,3] = 1
    	V[3,1] = 1
    	V[1,2] = 1
    	V[2,1] = 1
    elseif k==10
    	V[1,1] = -2
    	V[2,2] = -2
    	V[3,4] = -1
    	V[4,3] = -1
    	V[1,3] = 1
    	V[3,1] = 1
    	V[2,4] = 1
    	V[4,2] = 1
    	V[1,2] = 1
    	V[2,1] = 1
    else
       	print("Error, trying to calculate gradient for k=$k, outside 10 classes!".format(k))
    end

    return cn * V
end #get_cartan_matrix function

function CGMFR_get_w_o(k, cn, u_params, n)
	#Computing del W_pqrs / del O_ab
    wo = zeros(n, n, n, n, n, n)
    O = MF_real_unitary(n, u_params)
    lmbda_matrix = CGMFR_get_cartan_matrix(k, cn, n)
    delta = Diagonal(ones(n))
    @einsum wo[p,q,r,s,a,b] += delta[p,a] * O[q,b] * O[r,m] * O[s,m] * lmbda_matrix[b,m]
    @einsum wo[p,q,r,s,a,b] += delta[q,a] * O[p,b] * O[r,m] * O[s,m] * lmbda_matrix[b,m]
    @einsum wo[p,q,r,s,a,b] += delta[r,a] * O[p,l] * O[q,l] * O[s,b] * lmbda_matrix[l,b]
    @einsum wo[p,q,r,s,a,b] += delta[s,a] * O[p,l] * O[q,l] * O[r,b] * lmbda_matrix[l,b]

    return wo
end #get w_o

function MFR_get_o_angles(u_params, i, n)
	# returns the gradient w.r.t ith angles
	kappa = get_anti_symmetric(n)[i,:,:]
    K = construct_anti_symmetric(n, u_params)
    D, O = eigen(K)
	I = O' * kappa * O
    for a in 1:n
        for b in 1:n
            if abs(D[a] - D[b]) > 1e-8
                I[a, b] *= (exp(D[a] - D[b]) - 1) / (D[a] - D[b])
            end
        end
    end

    expD = Diagonal(exp.(D))
    
    return real.(O * I * expD * O')
end #get_o_angles

function CGMFR_MF_real_gradient_theta(k, x, diff, n = length(diff[:,1,1,1]))
    opnum = Int(n * (n - 1) / 2)

    wo = CGMFR_get_w_o(k, x[1], x[2:end], n)

    ograd = zeros(opnum)
    wtheta = zeros(n,n,n,n)
    for i in 1:opnum
        otheta = MFR_get_o_angles(x[2:end], i, n)
        @einsum wtheta[p,q,r,s] = wo[p,q,r,s,a,b] * otheta[a,b]
        ograd[i] = 2 * sum(diff .* wtheta)
    end
    
    return ograd
end

function CGMFR_MF_real_gradient_diag(k, x, diff, n, spin_orb=true)
	frag = fragment(x[2:end], x[1:1], k, n, spin_orb)
	tbt = fragment_to_normalized_tbt(frag, frag_flavour=CGMFR(), u_flavour=MF_real())

	return 2 * sum(diff .* tbt)
end

function CGMFR_MF_real_gradient(k, x, target, spin_orb=true, n = length(target[:,1,1,1]))
    frag = fragment(x[2:end], x[1:1], k, n, spin_orb)
    tbt = fragment_to_tbt(frag, frag_flavour=CGMFR(), u_flavour=MF_real())
    diff = tbt - target

    xgrad = zeros(length(x))

    xgrad[1] = CGMFR_MF_real_gradient_diag(k, x, diff, n, spin_orb)
    xgrad[2:end] = CGMFR_MF_real_gradient_theta(k, x, diff, n)
    
    return xgrad
end

function CGMFR_MF_real_gradient_full_rank(K_arr, x, target, spin_orb=true, n = length(target[:,1,1,1]))
	k_len = length(K_arr)
    u_num = Int(n*(n-1)/2)
    x_len = u_num + 1

    tbt = 0 .* target
    x_ini = 1
    for i in 1:k_len
    	frag = fragment(x[x_ini+1:x_ini+u_num], x[x_ini:x_ini], K_arr[i], n, spin_orb)
    	tbt += fragment_to_tbt(frag, frag_flavour=CGMFR(), u_flavour=MF_real())
    	x_ini += x_len
    end

    diff = tbt - target

    xgrad = zeros(length(x))

    x_ini = 1
    for k_num in 1:k_len
    	x0 = x[x_ini:x_ini+u_num]
        xgrad[x_ini] = CGMFR_MF_real_gradient_diag(K_arr[k_num], x0, diff, n, spin_orb)
        xgrad[x_ini+1:x_ini+u_num] = CGMFR_MF_real_gradient_theta(K_arr[k_num], x0, diff, n)
        x_ini += x_len
    end
    
    return xgrad
end

function parameter_gradient!(storage, x, target, class, n; spin_orb=true, frag_flavour=META.ff, u_flavour=META.uf)
	if typeof(u_flavour) == MF_real
		if typeof(frag_flavour) == CGMFR
			return storage .= CGMFR_MF_real_gradient(class, x, target, spin_orb, n)
		end
	end

	error("Gradient being calculated for frag_flavour=$frag_flavour and u_flavour=$u_flavour, not implemented!")
end

function full_rank_gradient!(storage, x, target, class_arr, n; spin_orb=true, frag_flavour=META.ff, u_flavour=META.uf)
	if typeof(u_flavour) == MF_real
		if typeof(frag_flavour) == CGMFR
			return storage .= CGMFR_MF_real_gradient_full_rank(class_arr, x, target, spin_orb, n)
		end
	end

	error("Gradient being calculated for frag_flavour=$frag_flavour and u_flavour=$u_flavour, not implemented!")
end