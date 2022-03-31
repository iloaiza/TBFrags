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
    
    return rotated_tbt
end

function obt_to_tbt_orbitals(obt)
    println("Warning, obt cannot be converted to tbt using orbitals: (ααββ) and (ββαα) are zero for obt but equal to (αααα) and (ββββ) for orbital tbt")
    n = size(obt)[1]
    D, U = eigen(obt)

    tbt_uu = zeros(2n,2n,2n,2n)
    tbt_dd = zeros(2n,2n,2n,2n)
    for i in 1:n
        α_spin = 2i-1
        β_spin = 2i
        tbt_uu[α_spin,α_spin,α_spin,α_spin] = D[i]
        tbt_dd[β_spin,β_spin,β_spin,β_spin] = D[i]
    end

    Uu = zeros(2n,2n)
    Ud = zeros(2n,2n)
    
    for i in 1:n
        for j in 1:n
            Uu[2i-1,2j-1] = U[i,j]
            Ud[2i,2j] = U[i,j]
        end
    end
    
    rotated_u = zeros(2n,2n,2n,2n)
    rotated_d = zeros(2n,2n,2n,2n)

    @einsum rotated_u[a,b,c,d] = Uu[a,l] * conj(Uu[b,l]) * Uu[c,l] * conj(Uu[d,l]) * tbt_uu[l,l,l,l]
    @einsum rotated_d[a,b,c,d] = Ud[a,l] * conj(Ud[b,l]) * Ud[c,l] * conj(Ud[d,l]) * tbt_dd[l,l,l,l]

    tbt_test = zeros(2n,2n,2n,2n)
    o_tbt = obt_to_tbt(obt)
    for i1 in 1:n
        for i2 in 1:n
            for i3 in 1:n
                for i4 in 1:n
                    tbt_test[2i1-1,2i2-1,2i3-1,2i4-1] = o_tbt[i1,i2,i3,i4]
                end
            end
        end
    end
    @show o_tbt
    @show tbt_test[1,1,:,:]
    @show tbt_test[3,4,:,:]
    @show tbt_uu[1,1,:,:]
    @show tbt_uu[3,4,:,:]
    @show sum(abs.(tbt_test - tbt_uu))

    tbt_so = tbt_uu + tbt_dd

    #=
    obt_so = zeros(2n,2n)
    for i in 1:n
        for j in 1:n
            obt_so[2i-1,2j-1] = obt[i,j]
            obt_so[2i,2j] = obt[i,j]
        end
    end
    
    tbt_so = obt_to_tbt(obt_so)
    # =#
    TBTuuuu = zeros(n,n,n,n)
    TBTuuud = zeros(n,n,n,n)
    TBTuudu = zeros(n,n,n,n)
    TBTuudd = zeros(n,n,n,n)
    TBTuduu = zeros(n,n,n,n)
    TBTudud = zeros(n,n,n,n)
    TBTuddu = zeros(n,n,n,n)
    TBTuddd = zeros(n,n,n,n)
    TBTduuu = zeros(n,n,n,n)
    TBTduud = zeros(n,n,n,n)
    TBTdudu = zeros(n,n,n,n)
    TBTdudd = zeros(n,n,n,n)
    TBTdduu = zeros(n,n,n,n)
    TBTddud = zeros(n,n,n,n)
    TBTdddu = zeros(n,n,n,n)
    TBTdddd = zeros(n,n,n,n)

    for i1 in 1:n
        for i2 in 1:n
            for i3 in 1:n
                for i4 in 1:n
                    TBTuuuu[i1,i2,i3,i4] = tbt_so[2i1-1,2i2-1,2i3-1,2i4-1]
                    TBTuuud[i1,i2,i3,i4] = tbt_so[2i1-1,2i2-1,2i3-1,2i4]
                    TBTuudu[i1,i2,i3,i4] = tbt_so[2i1-1,2i2-1,2i3,2i4-1]
                    TBTuudd[i1,i2,i3,i4] = tbt_so[2i1-1,2i2-1,2i3,2i4]
                    TBTuduu[i1,i2,i3,i4] = tbt_so[2i1-1,2i2,2i3-1,2i4-1]
                    TBTudud[i1,i2,i3,i4] = tbt_so[2i1-1,2i2,2i3-1,2i4]
                    TBTuddu[i1,i2,i3,i4] = tbt_so[2i1-1,2i2,2i3,2i4-1]
                    TBTuddd[i1,i2,i3,i4] = tbt_so[2i1-1,2i2,2i3,2i4]
                    TBTduuu[i1,i2,i3,i4] = tbt_so[2i1,2i2-1,2i3-1,2i4-1]
                    TBTduud[i1,i2,i3,i4] = tbt_so[2i1,2i2-1,2i3-1,2i4]
                    TBTdudu[i1,i2,i3,i4] = tbt_so[2i1,2i2-1,2i3,2i4-1]
                    TBTdudd[i1,i2,i3,i4] = tbt_so[2i1,2i2-1,2i3,2i4]
                    TBTdduu[i1,i2,i3,i4] = tbt_so[2i1,2i2,2i3-1,2i4-1]
                    TBTddud[i1,i2,i3,i4] = tbt_so[2i1,2i2,2i3-1,2i4]
                    TBTdddu[i1,i2,i3,i4] = tbt_so[2i1,2i2,2i3,2i4-1]
                    TBTdddd[i1,i2,i3,i4] = tbt_so[2i1,2i2,2i3,2i4]
                end
            end
        end
    end

    @show sum(abs.(TBTuuuu))
    @show sum(abs.(TBTuuud))
    @show sum(abs.(TBTuudu))
    @show sum(abs.(TBTuudd))
    @show sum(abs.(TBTuduu))
    @show sum(abs.(TBTudud))
    @show sum(abs.(TBTuddu))
    @show sum(abs.(TBTuddd))
    @show sum(abs.(TBTduuu))
    @show sum(abs.(TBTduud))
    @show sum(abs.(TBTdudu))
    @show sum(abs.(TBTdudd))
    @show sum(abs.(TBTdduu))
    @show sum(abs.(TBTddud))
    @show sum(abs.(TBTdddu))
    @show sum(abs.(TBTdddd))

    return tbt_so
end