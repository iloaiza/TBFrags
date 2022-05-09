using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=ham_name]
args_len = length(ARGS)
const ham_name = ARGS[1]

@everywhere include("include.jl")

println("Loading calculations with:")
@show ham_name
if ham_name == "li" || ham_name == "reiher"
	h5name = DATAFOLDER * "eri_$(ham_name).h5"
	tbt_mo = h5read(h5name,"eri")/2 #to be consistent with 1/2 factor
	n = size(tbt_mo)[1]
	obt_mo = h5read(h5name,"h0") - sum([tbt_mo[:,k,k,:] for k in 1:n])
	tbt_mo_tup = (obt_mo, tbt_mo)
else
	tbt_mo_tup, h_ferm, num_elecs = obtain_SD(ham_name, basis=basis, ferm=true, spin_orb=false, geometry=geometry, n_elec=true)
end

println("Starting qubit treatment...")
#=
println("Performing fermion to qubit mapping:")
@time H_full_q = qubit_transform(h_ferm, "jw")
#@show H_full_q
println("\n\n\n Qubit treatment of Hamiltonian:")
qubit_treatment(H_full_q)
# = #
println("Starting T' boosted AC-SI routine")
@time L1,num_ops = bin_anticommuting_jw_sorted_insertion(tbt_mo_tup[1], tbt_mo_tup[2], cutoff = 1e-6)
println("BOOSTED AC-SI L1=$(L1)($(ceil(log2(num_ops))))\n\n\n")
# =#

tol = 1e-6

println("\n\n\n Starting SVD routine for separated 1 and 2-body terms with cutoff tolerance $tol")
@time SVD_CARTAN_TBTS_2B, SVD_TBTS_2B = tbt_svd(tbt_mo_tup[2], tol=tol, spin_orb=false, ret_op=false)

α_SVD_2B = size(SVD_TBTS_2B)[1]
@show α_SVD_2B

println("Strarting L1 calculations")
@time SVD_L1_2B, SVD_E_RANGES_2B= L1_frags_treatment(SVD_CARTAN_TBTS_2B, false)

println("One-body inclusion for modified one-body term")
obt_mo_mod = tbt_mo_tup[1] + 2*sum([tbt_mo_tup[2][:,:,r,r] for r in 1:size(tbt_mo_tup[1])[1]]) 
D,U = eigen(obt_mo_mod)
SVD_L1_2B_MOD = collect(SVD_L1_2B)
push!(SVD_L1_2B_MOD, cartan_obt_l1_cost(D, false))
@show SVD_L1_2B_MOD
RANGES = zeros(1,2)
obt_so = obt_orb_to_so(Diagonal(D))
RANGES[:] = CSA_obt_range(obt_so)
SVD_E_RANGES_2B_MOD = vcat(SVD_E_RANGES_2B, RANGES)
ΔE_SVD_2B_MOD = [(SVD_E_RANGES_2B_MOD[i,2] - SVD_E_RANGES_2B_MOD[i,1])/2 for i in 1:α_SVD_2B]
@show ΔE_SVD_2B_MOD

println("CSA L1 bounds (NR) (SVD 1(mod)-2):")
@show sum(SVD_L1_2B_MOD)/2
println("Shifted minimal norm (SR) (SVD 1(mod)-2):")
@show sum(ΔE_SVD_2B_MOD)

println("\n\n Including vanilla one-body part")
D,U = eigen(tbt_mo_tup[1])
SVD_L1_2B = collect(SVD_L1_2B)
push!(SVD_L1_2B, cartan_obt_l1_cost(D, false))
@show SVD_L1_2B
RANGES = zeros(1,2)
obt_so = obt_orb_to_so(Diagonal(D))
RANGES[:] = CSA_obt_range(obt_so)
SVD_E_RANGES_2B = vcat(SVD_E_RANGES_2B, RANGES)
ΔE_SVD = [(SVD_E_RANGES_2B[i,2] - SVD_E_RANGES_2B[i,1])/2 for i in 1:α_SVD_2B]
@show ΔE_SVD

println("CSA L1 bounds (NR) (SVD 1-2):")
@show sum(SVD_L1_2B)/2
println("Shifted minimal norm (SR) (SVD 1-2):")
@show sum(ΔE_SVD)

println("\n\n\n Starting SVD routine for 1+2 body terms with cutoff tolerance $tol:")
println("Transforming spacial orbital tensors to spin-orbitals")
@time tbt_so = tbt_to_so(tbt_mo_tup, false)
@time SVD_CARTAN_TBTS, SVD_TBTS = tbt_svd(tbt_so, tol=tol, spin_orb=true, ret_op=false)
α_SVD = size(SVD_TBTS)[1]
@show α_SVD

println("Starting L1 calculations")
@time SVD_L1, SVD_E_RANGES = L1_frags_treatment(SVD_CARTAN_TBTS, true)
@show SVD_L1
ΔE_SVD = [(SVD_E_RANGES[i,2] - SVD_E_RANGES[i,1])/2 for i in 1:α_SVD]
@show ΔE_SVD

println("CSA L1 bounds (NR) (SVD 1+2):")
@show sum(SVD_L1)/2
println("Shifted minimal norm (SR) (SVD 1+2):")
@show sum(ΔE_SVD)

println("\n\n\n Starting SVD routine for 1+2 body terms with cutoff tolerance $tol, using modified one-body operator:")
println("Transforming spacial orbital tensors to spin-orbitals")
@time tbt_so = tbt_to_so((obt_mo_mod, tbt_mo_tup[2]), false)
@time SVD_CARTAN_TBTS, SVD_TBTS = tbt_svd(tbt_so, tol=tol, spin_orb=true, ret_op=false)
α_SVD = size(SVD_TBTS)[1]
@show α_SVD

println("Starting L1 calculations")
@time SVD_L1, SVD_E_RANGES = L1_frags_treatment(SVD_CARTAN_TBTS, true)
@show SVD_L1
ΔE_SVD = [(SVD_E_RANGES[i,2] - SVD_E_RANGES[i,1])/2 for i in 1:α_SVD]
@show ΔE_SVD

println("CSA L1 bounds (NR) (SVD 1(mod)+2):")
@show sum(SVD_L1)/2
println("Shifted minimal norm (SR) (SVD 1(mod)+2):")
@show sum(ΔE_SVD)