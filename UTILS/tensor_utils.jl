function obt_to_tbt(obt)
	## transform one-body tensor into two-body tensor
    ## does not work for orbitals! (i.e. spin_orb = false)
    
    #println("Transforming one-body tensor into two-body object, small numerical errors might appear...")
	
    Dobt, Uobt = eigen(obt)
    #obt ≡ Uobt * Diagonal(Dobt) * (Uobt')

    n = size(obt)[1]

    tbt = zeros(n,n,n,n)
    for i in 1:n
        tbt[i,i,i,i] = Dobt[i]
    end

    rotated_tbt = zeros(n,n,n,n)

    @einsum rotated_tbt[a,b,c,d] = Uobt[a,l] * conj(Uobt[b,l]) * Uobt[c,l] * conj(Uobt[d,l]) * tbt[l,l,l,l]
    
    return rotated_tbt
end

function tbt_orb_to_so(tbt)
    n = size(tbt)[1]
    n_qubit = 2n

    tbt_so = zeros(2n,2n,2n,2n)
    for i1 in 1:n
        for i2 in 1:n
            for i3 in 1:n
                for i4 in 1:n
                    for a in -1:0
                        for b in -1:0
                            tbt_so[2i1+a,2i2+a,2i3+b,2i4+b] = tbt[i1,i2,i3,i4]
                        end
                    end
                end
            end
        end
    end

    return tbt_so
end

function obt_orb_to_so(obt)
    n = size(obt)[1]
    n_qubit = 2n

    obt_so = zeros(2n,2n)
    for i1 in 1:n
        for i2 in 1:n
            for a in -1:0
                obt_so[2i1+a,2i2+a] = obt[i1,i2]
            end
        end
    end

    return obt_so
end

function cartan_obt_to_tbt(obt, tiny=1e-15)
    #transform a Cartan one-body tensor into a two-body tensor
    if sum(abs.(Diagonal(obt) - obt)) > tiny
        error("Warning, trying to transform non-diagonal Cartan obt to tbt!")
    end
    n = size(obt)[1]

    tbt = zeros(n,n,n,n)
    for i in 1:n
        tbt[i,i,i,i] = obt[i,i]
    end

    return tbt
end

function cartan_tbt_to_triang(tbt, n=size(tbt)[1])
    # transforms cartan polynomial tbt into triangular form (i.e. tbt[i,i,j,j] for i≤j)
    tbt_tri = zeros(n,n,n,n)

    for i in 1:n
        tbt_tri[i,i,i,i] = tbt[i,i,i,i]
    end

    for i in 1:n
        for j in i+1:n
            tbt_tri[i,i,j,j] = tbt[i,i,j,j] + tbt[j,j,i,i]
        end
    end

    return tbt_tri
end

function tbt_to_so(tbt :: Array, spin_orb)
    #return two-body tensor in spin-orbitals
    if spin_orb == false
        return tbt_orb_to_so(tbt)
    else
        return tbt
    end
end

function tbt_to_so(tbt :: Tuple, spin_orb)
    #return two-body tensor in spin-orbitals
    if spin_orb == false
        return tbt_orb_to_so(tbt[2]) + obt_to_tbt(obt_orb_to_so(tbt[1]))
    else
        return tbt[2] + obt_to_tbt(tbt[1])
    end
end
