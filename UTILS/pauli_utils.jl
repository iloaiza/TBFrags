struct pauli_bit
    bin :: Tuple{Bool,Bool}
end

function pauli_bit(i :: Int64)
    if i == 0
        return pauli_bit((0,0))
    elseif i == 1
        return pauli_bit((1,0))
    elseif i == 2
        return pauli_bit((1,1))
    elseif i == 3
        return pauli_bit((0,1))
    end
end

struct pauli_word
    bits :: Array{pauli_bit,1}
    size :: Int64
    coeff :: Number
end

function pauli_word(bits, coeff=1.0)
    pauli_size = length(bits)

    return pauli_word(bits,length(bits),coeff)
end

function pauli_num_from_bit(p :: pauli_bit)
    if p.bin[1] == false && p.bin[2] == false
        return 0
    elseif p.bin[1] == true && p.bin[2] == false
        return 1
    elseif p.bin[1] == true && p.bin[2] == true
        return 2
    elseif p.bin[1] == false && p.bin[2] == true
        return 3
    end
end

function ϵ(i,j)
    #return what k corresponds for Pauli multiplication and sign
    if i == j
        return 1, 0
    elseif i == 0
        return 1, j
    elseif j == 0
        return 1, i
    else
        if i == 1 && j == 2
            return 1im, 3
        elseif i == 2 && j == 1
            return -1im, 3
        elseif i == 2 && j == 3
            return 1im, 1
        elseif i == 3 && j == 2
            return -1im, 1
        elseif i == 3 && j == 1
            return 1im, 2
        elseif i == 1 && j == 3
            return -1im, 2
        end
    end
end


function bit_sum(b1 :: pauli_bit, b2 :: pauli_bit)
    #σi*σj
    i = pauli_num_from_bit(b1)
    j = pauli_num_from_bit(b2)
    phase, k = ϵ(i, j)

    return pauli_bit(k), phase
end

import Base.*

function *(a :: pauli_word, b :: pauli_word)
    len1 = a.size
    len2 = b.size

    if len1 != len2
        error("Unequal lengths!")
    end

    bits = pauli_bit[]
    sizehint!(bits, len1)

    tot_phase = 1 + 0im
    for i in 1:len1 
        prod_result, phase = bit_sum(a.bits[i], b.bits[i])
        push!(bits, prod_result)
        tot_phase *= phase
    end

    coeff = a.coeff * b.coeff * tot_phase

    return pauli_word(bits, coeff)
end

import Base.zero

zero(T::Type{pauli_bit}) = pauli_bit((0,0))

function bin_vec_to_pw(b_vec, n_qubits = Int(size(b_vec)[1]/2), coeff=1)
    bits = zeros(pauli_bit, n_qubits)

    for i in 1:n_qubits
        bits[i] = pauli_bit((b_vec[i], b_vec[n_qubits+i]))
    end

    return pauli_word(bits, coeff)
end

function pw_to_bin_vec(pw)
    n_qubits = pw.size
    b_vec = zeros(Bool, 2*n_qubits)

    for (i,bit) in enumerate(pw.bits)
        b_vec[i] = bit.bin[1]
        b_vec[i+n_qubits] = bit.bin[2]
    end

    return b_vec
end

function Qpqσ_bin_vec(p,q,σ,n_qubits,so_consecutive=true,n_mos = Int(n_qubits/2))
    #return symplectic vector corresponding to Qpqσ Pauli word and phase for multiplication result
    #σ = 0(1) for spin α(β)
    bin_vec = zeros(Bool,2*n_qubits)
 
    if so_consecutive
        if p < q
            # xpσ*vec(z)*xqσ
            phase = 1
            bin_vec[2p-1+σ] = true
            bin_vec[2q-1+σ] = true
            for i in 2p+σ:2q-2+σ
                #bin_vec[2i-1+σ+n_qubits] = true
                bin_vec[i+n_qubits] = true
            end
        elseif p > q
            phase = 1
            # yqσ*vec(z)*yqσ
            bin_vec[2p-1+σ] = true
            bin_vec[2p-1+σ+n_qubits] = true
            bin_vec[2q-1+σ] = true
            bin_vec[2q-1+σ+n_qubits] = true
            for i in 2q+σ:2p-2+σ
                #bin_vec[2i-1+σ+n_qubits] = true
                bin_vec[i+n_qubits] = true
            end
        else
            # - zpσ
            phase = -1
            bin_vec[2p-1+σ+n_qubits] = true
        end
    else
        if p < q
            # xpσ*vec(z)*xqσ
            phase = 1
            bin_vec[p+σ*n_mos] = true
            bin_vec[q+σ*n_mos] = true
            for i in p+1:q-1
                bin_vec[i+σ*n_mos+n_qubits] = true
            end
        elseif p > q
            phase = 1
            # yqσ*vec(z)*yqσ
            bin_vec[p+σ*n_mos] = true
            bin_vec[p+σ*n_mos+n_qubits] = true
            bin_vec[q+σ*n_mos] = true
            bin_vec[q+σ*n_mos+n_qubits] = true
            for i in q+1:p-1
                bin_vec[i+σ*n_mos+n_qubits] = true
            end
        else
            # - zpσ
            phase = -1
            bin_vec[p+σ*n_mos+n_qubits] = true
        end
    end
   
    #@show p,q,σ, phase * qbit.binary_vector_to_pauli_word(Int.(bin_vec))
    
    return bin_vec, phase
end

function bin_vec_prod(bin1,bin2,n_qubits=Int(length(bin1)/2))
    #returns, up to a phase, pauli corresponding to bin1*bin2
    bin3 = zeros(Bool,2*n_qubits)
    bin3 .= (bin1 + bin2) .% 2
end

function are_Qs_anticommuting(p,q,α,r,s,β,n_qubits)
    #if α != β
    #    return false
    #else
        bin_pq,_ = Qpqσ_bin_vec(p,q,α,n_qubits)
        bin_rs,_ = Qpqσ_bin_vec(r,s,β,n_qubits)
        return Bool(binary_is_anticommuting(bin_pq,bin_rs, n_qubits))
    #end
end

zero(T::Type{Tuple{Int64,Int64}}) = (0,0)
zero(T::Type{Tuple{Int64,Int64,Int64,Int64}}) = (0,0,0,0)

function ord_to_pq(ord_ind, is_tbt, n)
    D = Dict{Int64,Tuple}()
    n2 = n^2

    pq_arr = zeros(Tuple{Int64,Int64},n,n)
    pqrs_arr = zeros(Tuple{Int64,Int64,Int64,Int64},n,n,n,n)

    for p in 1:n
        for q in 1:n
            pq_arr[p,q] = (p,q)
            for r in 1:n
                for s in 1:n
                    pqrs_arr[p,q,r,s] = (p,q,r,s)
                end
            end
        end
    end

    for i in 1:length(ord_ind)
        i_eff = ord_ind[i]
        if is_tbt[i] == true
            i_eff = ord_ind[i] - n2
            get!(D,i,pqrs_arr[i_eff])
        else
            get!(D,i,pq_arr[i_eff])
        end
    end

    return D
end

function bin_array_reducer(bin_vecs)
    #bin_vecs = zeros(Bool,pnum,2*n_qubits)
    #returns same array over reduced pnumber over paulis which are repeated (does not consider coefficients)
    pnum = size(bin_vecs)[1]
    #n_qubits = Int(size(bin_vecs)[2]/2)

    groups = Array{Int64,1}[[]] #first group holds identities
    sizehint!(groups, pnum)
    group_reps = Int64[]
    sizehint!(group_reps, pnum)
    t00 = time()
    @show pnum
    for i in 1:pnum
        if i%10000 == 0
            @show i
            println("Time since last check: $(time() - t00)")
            t00 = time()
        end
        #@show groups
        if sum(bin_vecs[i,:]) == 0
            push!(groups[1],i)
            #println("Added $i to groups[1]:")
            #@show bin_vecs[i,:]
        else
            #println("Sum is not 0 for i=$i")
            is_grouped = false
            for (group_num,i_group) in enumerate(group_reps)
                if bin_vecs[i,:] == bin_vecs[i_group,:]
                    push!(groups[group_num+1], i)
                    is_grouped = true
                    break
                end
            end

            if is_grouped == false
                #println("$i is not grouped, creating group!")
                push!(groups,[i])
                push!(group_reps, i)
            end
        end
    end

    num_paulis = length(group_reps)
    #@show group_reps

    return bin_vecs[group_reps,:], groups
end

function bin_ac_si(bin_vecs, vals_ord, pnum=size(bin_vecs)[1], n_qubits=Int(size(bin_vecs)[2]/2))
    #sorted insertion algorithm over binary vectors array
    is_grouped = zeros(Bool,pnum)
    group_arrs = Array{Int64,1}[]
    vals_arrs = Array{Complex{Float64},1}[]

    println("Running sorted insertion algorithm")
    @show sum(abs.(vals_ord))
    for i in 1:pnum
        if is_grouped[i] == false
            curr_group = [i]
            curr_vals = [vals_ord[i]]
            is_grouped[i] = true
            for j in i+1:pnum
                if is_grouped[j] == false
                    if binary_is_anticommuting(bin_vecs[i,:],bin_vecs[j,:], n_qubits) == 1
                        antic_w_group = true
                        for k in curr_group[2:end]
                            if binary_is_anticommuting(bin_vecs[k,:],bin_vecs[j,:], n_qubits) == 0
                                antic_w_group = false
                                break
                            end
                        end

                        if antic_w_group == true
                            push!(curr_group,j)
                            push!(curr_vals,vals_ord[j])
                            is_grouped[j] = true
                        end
                    end
                end
            end
            push!(group_arrs,curr_group)
            push!(vals_arrs, curr_vals)
        end
    end

    if prod(is_grouped) == 0
        println("Error, not all terms are grouped after AC-SI algorithm!")
        @show is_grouped
    end

    num_groups = length(group_arrs)
    group_L1 = zeros(num_groups)
    for i in 1:num_groups
        for val in vals_arrs[i]
            group_L1[i] += abs2(val)
        end
    end

    L1_norm = sum(sqrt.(group_L1))
    return L1_norm, num_groups
end

function bin_anticommuting_jw_sorted_insertion(obt_mo, tbt_mo; cutoff = 1e-6)
    n = size(obt_mo)[1]
    n_qubits = 2n

    λV = sum(abs.(tbt_mo))
    @show λV
    println("Naive λT = $(sum(abs.(obt_mo)))")
    obt_tilde = obt_mo + 2*sum([tbt_mo[:,:,r,r] for r in 1:n])
    λT = sum(abs.(obt_tilde))
    println("Tilde λT = $λT")
    @show λV + λT

    println("Sorting by coefficients")
    all_coeffs = append!([obt_tilde...],[tbt_mo...])
    ord_ind = sortperm(abs.(all_coeffs))[end:-1:1]
    ferm_vals_ord = all_coeffs[ord_ind]
    global i_fin = length(ferm_vals_ord)
    for (i,val) in enumerate(ferm_vals_ord)
        if abs(val) < cutoff
            global i_fin = i-1
            println("Using first $i fermionic coefficients out of $(n^2+n^4) total number, cutoff value is $cutoff for coefficient value $val")
            break
        end
    end
    ferm_vals_ord = ferm_vals_ord[1:i_fin]
    ord_ind = ord_ind[1:i_fin]
    is_tbt = append!(zeros(Bool,n^2),ones(Bool,n^4))[ord_ind]
    pq_dict = ord_to_pq(ord_ind, is_tbt, n)
    
    pnum_ferm = 0
    for i in 1:i_fin #over molecular orbitals, one-body tensors mapped to 2, two-body to 4
        if is_tbt[i] == true
            pnum_ferm += 4
        else
            pnum_ferm += 2
        end
    end
    @show pnum_ferm


    vals_ord_all = SharedArray(zeros(Complex{Float64},pnum_ferm))
    init_i_arr = zeros(Int64,i_fin) #i to what number of bin vector it corresponds to
    init_i_arr[1] = 1
    if is_tbt[1] == true
        for α in 0:1
            for β in 0:1
                vals_ord_all[1+2α+β] = ferm_vals_ord[1]/4
            end
        end
    else
        for α in 0:1
            vals_ord_all[1+α] = ferm_vals_ord[1]/2
        end
    end

    for i in 2:i_fin
        if is_tbt[i-1] == true
            init_i_arr[i] = init_i_arr[i-1] + 4
            for α in 0:1
                for β in 0:1
                    vals_ord_all[init_i_arr[i]+2α+β] = ferm_vals_ord[i]/4
                end
            end
        else
            init_i_arr[i] = init_i_arr[i-1] + 2
            for α in 0:1
                vals_ord_all[init_i_arr[i]+α] = ferm_vals_ord[i]/2
            end
        end
    end
    
    println("Building binary vectors")
    t00 = time()
    bin_vecs = SharedArray(zeros(Bool,pnum_ferm,2*n_qubits))
    @sync @distributed for i in 1:length(ord_ind)
        ind = ord_ind[i]
        if is_tbt[i] == false
            for α in 0:1
                p,q = pq_dict[i]
                bin_vecs[init_i_arr[i]+α,:], phase = Qpqσ_bin_vec(p,q,α,n_qubits)
                vals_ord_all[init_i_arr[i]+α] *= phase
            end
        else
            for α in 0:1
                for β in 0:1
                    p,q,r,s = pq_dict[i]
                    qpa, phase1 = Qpqσ_bin_vec(p,q,α,n_qubits)
                    qrb, phase2 = Qpqσ_bin_vec(r,s,β,n_qubits)
                    pw1 = bin_vec_to_pw(qpa, n_qubits, phase1)
                    pw2 = bin_vec_to_pw(qrb, n_qubits, phase2)
                    pw_prod = pw1*pw2
                    bin_vecs[init_i_arr[i]+2α+β,:] = pw_to_bin_vec(pw_prod)
                    vals_ord_all[init_i_arr[i]+2α+β] *= pw_prod.coeff
                end
            end
        end
    end
    println("Took $(time()-t00) seconds to run vector building routine")
    #@time bin_vecs = collect(bin_vecs)

    println("Reducing array of binary vectors")
    @time bin_vecs, degenerate_groups = bin_array_reducer(bin_vecs)
    pnum = size(bin_vecs)[1]
    @show pnum
    vals_ord = zeros(Complex{Float64}, pnum)
    
    for i in 1:pnum
        vals_ord[i] = sum(vals_ord_all[degenerate_groups[i+1]])
    end
    @show sum(abs.(vals_ord))
    @show sum(abs.(vals_ord_all))
    
    # =
    tot_op = of.QubitOperator.zero()
    for i in 1:pnum
        tot_op += vals_ord[i] * qbit.binary_vector_to_pauli_word(Int.(bin_vecs[i, :]))
    end
    
    #@show tot_op
    #@show qubit_transform(tbt_to_ferm((obt_mo, tbt_mo),false))
    @show tot_op - qubit_transform(tbt_to_ferm((obt_mo, tbt_mo),false))
    # =#

    return bin_ac_si(bin_vecs, vals_ord)
end
