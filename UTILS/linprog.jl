import HiGHS
import Ipopt

function cartan_tbt_to_vec(cartan_so)
    n = size(cartan_so)[1]

    ν_len = Int(n*(n+1)/2)
    λ_vec = zeros(ν_len)
    idx = 0
    for i in 1:n
        for j in 1:i
            idx += 1
            if i == j
                λ_vec[idx] = cartan_so[i,i,i,i]
            else
                λ_vec[idx] = cartan_so[i,i,j,j] + cartan_so[j,j,i,i]
            end
        end
    end
    
    return λ_vec
end

function τ_mat_builder(SYM_ARR)
    n = size(SYM_ARR[1])[1]
    ν_len = Int(n*(n+1)/2)
    num_syms = length(SYM_ARR)
    τ_mat = zeros(ν_len,num_syms)
    
    for i in 1:num_syms
        τ_mat[:,i] = cartan_tbt_to_vec(SYM_ARR[i])
    end

    return τ_mat
end

function L1_linprog_optimizer_frag(cartan_so, τ_mat, verbose=false, model="highs")
    if model == "highs"
        L1_OPT = Model(HiGHS.Optimizer)
    elseif model == "ipopt"
        L1_OPT = Model(Ipopt.Optimizer)
    else
        error("Not defined for model = $model")
    end
    
    if verbose == false
        set_silent(L1_OPT)
    end

    λ_vec = cartan_tbt_to_vec(cartan_so)
    ν_len,num_syms = size(τ_mat)

    @variables(L1_OPT, begin
        s[1:num_syms]
        t[1:ν_len]
    end)

    @objective(L1_OPT, Min, sum(t))

    @constraint(L1_OPT, low, τ_mat*s - t - λ_vec .<= 0)
    @constraint(L1_OPT, high, τ_mat*s + t - λ_vec .>= 0)

    optimize!(L1_OPT)

    return value.(s)
end

function L1_linprog_optimizer_full(tbt_mo_tup, SYM_ARR, verbose=false)
    #input: 1- and 2-body tuple in spatial orbitals
    tbt_full = tbt_to_so(tbt_mo_tup, false)

    n = size(tbt_full)[1]
    N4 = n^4

    tbt_full = reshape(tbt_full, N4)
    
    num_syms = length(SYM_ARR)

    syms_mat = zeros(N4, num_syms)
    for i in 1:num_syms
        syms_mat[:,i] = reshape(SYM_ARR[i],N4)
    end

    L1_OPT = Model(Ipopt.Optimizer)
    
    if verbose == false
        set_silent(L1_OPT)
    end

    @variables(L1_OPT, begin
        s[1:num_syms]
        t[1:N4]
    end)

    @objective(L1_OPT, Min, sum(t))

    @constraint(L1_OPT, low, syms_mat*s - t - tbt_full .<= 0)
    @constraint(L1_OPT, high, syms_mat*s + t - tbt_full .>= 0)

    optimize!(L1_OPT)

    return value.(s)
    
end
