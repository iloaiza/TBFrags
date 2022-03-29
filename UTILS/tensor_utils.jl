function obt_to_tbt(obt)
	#transform one-body tensor into two-body tensor
    #warning, does not work for orbitals! (i.e. spin_orb = false)
    println("Transforming one-body tensor into two-body object, small numerical errors might appear...")
	
    Dobt, Uobt = eigen(obt)
    #obt ≡ Uobt * Diagonal(Dobt) * (Uobt')

    n = size(obt)[1]

    tbt = zeros(n,n,n,n)
    for i in 1:n
        tbt[i,i,i,i] = Dobt[i]
    end

    rotated_tbt = zeros(n,n,n,n)

    @einsum rotated_tbt[a,b,c,d] = Uobt[a,l] * conj(Uobt[b,l]) * Uobt[c,l] * conj(Uobt[d,l]) * tbt[l,l,l,l]
    #@einsum rotated_tbt[a,b,c,d] = Uobt[a,l] * Uobt[b,l] * Uobt[c,m] * Uobt[d,m] * tbt[l,l,m,m]

    return rotated_tbt
end