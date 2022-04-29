using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=ham_name]
args_len = length(ARGS)
const ham_name = ARGS[1]

@everywhere include("include.jl")

println("Loading calculations with:")
@show ham_name

tbt_mo_tup, tbt_so, h_ferm = load_full_ham_tbt(ham_name, spin_orb=false, prefix=".")

println("Starting qubit treatment...")
n = size(tbt_so)[1]
n_qubit = n
println("Performing fermion to qubit mapping:")
@time H_full_q = qubit_transform(h_ferm)

println("Qubit treatment of Hamiltonian:")
qubit_treatment(H_full_q)

tol = 1e-6
println("Starting SVD routine with cutoff tolerance $tol:")
@time SVD_CARTAN_TBTS, SVD_TBTS = tbt_svd(tbt_so, tol=tol, spin_orb=true)
α_SVD = size(SVD_TBTS)[1]
@show α_SVD

println("Starting L1 calculations")
@time SVD_L1, SVD_E_RANGES,_ = L1_frags_treatment(SVD_TBTS, SVD_CARTAN_TBTS, true)

println("CSA L1 bounds (NR):")
@show sum(SVD_L1)/2
println("Shifted minimal norm (SR):")
ΔE_SVD = [(SVD_E_RANGES[i,2] - SVD_E_RANGES[i,1])/2 for i in 1:α_SVD]
@show sum(ΔE_SVD)

println("Starting SVD routine for separated 1 and 2-body terms")
@time SVD_CARTAN_TBTS_2B, SVD_TBTS_2B = tbt_svd(tbt_mo_tup[2], tol=tol, spin_orb=false)

α_SVD_2B = size(SVD_TBTS_2B)[1]
@show α_SVD_2B

println("Strarting L1 calculations")
@time SVD_L1_2B, SVD_E_RANGES_2B,_ = L1_frags_treatment(SVD_TBTS_2B, SVD_CARTAN_TBTS_2B, false)

D,U = eigen(tbt_mo_tup[1])
push!(SVD_L1_2B, cartan_obt_l1_cost(Diagonal(D), false))
RANGES = zeros(1,2)
obt_so = obt_orb_to_so(Diagonal(D))
RANGES[:] = CSA_obt_range(obt_so)
SVD_E_RANGES_2B = vcat(SVD_E_RANGES_2B, RANGES)

println("CSA L1 bounds (NR) (SVD 1-2):")
@show sum(SVD_L1_2B)/2
println("Shifted minimal norm (SR) (SVD 1-2):")
ΔE_SVD = [(SVD_E_RANGES_2B[i,2] - SVD_E_RANGES_2B[i,1])/2 for i in 1:α_SVD_2B]
@show sum(ΔE_SVD)