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
	h_ferm = tbt_to_ferm(tbt_mo_tup, false)
else
	tbt_mo_tup, h_ferm, num_elecs = obtain_SD(ham_name, basis=basis, ferm=true, spin_orb=false, geometry=geometry, n_elec=true)
end

println("Starting qubit treatment...")
# =
println("Performing fermion to qubit mapping:")
@time H_full_q = qubit_transform(h_ferm, "jw")
#@show H_full_q
println("\n\n\n Qubit treatment of Hamiltonian:")
qubit_treatment(H_full_q)
#= #
println("Starting T' boosted AC-SI routine")
@time L1,num_ops = bin_anticommuting_jw_sorted_insertion(tbt_mo_tup[1], tbt_mo_tup[2], cutoff = 1e-6)
println("BOOSTED AC-SI L1=$(L1)($(ceil(log2(num_ops))))\n\n\n")
# =# # =#

SVD_tol = 1e-6

println("\n\n\n Starting SVD routine for separated 1 and 2-body terms with cutoff tolerance $SVD_tol")
@time NR_2B, SR_2B = svd_optimized(tbt_mo_tup[2], tol=SVD_tol, spin_orb=false)
D,U = eigen(tbt_mo_tup[1])
RANGES = zeros(1,2)
obt_so = obt_orb_to_so(Diagonal(D))
RANGES[:] = CSA_obt_range(obt_so)
SR_2B = vcat(SR_2B, RANGES)
push!(NR_2B, cartan_obt_l1_cost(D, false))
ΔE_2B = [(SR_2B[i,2] - SR_2B[i,1])/2 for i in 1:length(NR_2B)]

println("CSA L1 bounds (NR) (SVD 1-2):")
@show sum(NR_2B)/2

println("Shifted minimal norm (SR) (SVD 1-2):")
@show sum(ΔE_2B)

println("\n\n\n Starting df routine...")
@time svd_optimized_df(tbt_mo_tup, tol=1e-6, tiny=1e-8)

println("Starting SVD routine for separated terms with Google's grouping technique:")
CARTANS, TBTS = tbt_svd(tbt_mo_tup[2], tol=SVD_tol, spin_orb=false, ret_op=false)
α_SVD = size(CARTANS)[1]
n = size(tbt_mo_tup[1])[1]
obt_tilde = tbt_mo_tup[1] + 2*sum([tbt_mo_tup[2][:,:,r,r] for r in 1:n])
obt_D, _ = eigen(obt_tilde)
λT = sum(abs.(obt_D))

global λV = 0.0
for i in 1:α_SVD
	global λV += cartan_to_qubit_l1_treatment(CARTANS[i,:,:,:,:], false)
end
@show λT, λV, λT+λV

println("\n\n\n STARTING SYMMETRY OPTIMIZATIONS ROUTINE: BEFORE")
tbt_so = tbt_to_so(tbt_mo_tup[2], false)
tbt_ham_opt, x_opt = symmetry_cuadratic_optimization(tbt_so, true, S2=false)
CARTANS, TBTS = tbt_svd(tbt_ham_opt, tol=SVD_tol, spin_orb=true, ret_op=false)
α_SVD = size(CARTANS)[1]
n = size(tbt_ham_opt)[1]
#obt_tilde_sym_b4 = obt_orb_to_so(obt_tilde)
obt_tilde_sym_b4 = obt_orb_to_so(tbt_mo_tup[1]) + 2*sum([tbt_ham_opt[:,:,r,r] for r in 1:n])
#obt_tilde_sym_b4, _ = one_body_symmetry_cuadratic_optimization(obt_tilde_sym_b4, true)
obt_D, _ = eigen(obt_tilde_sym_b4)
λT = sum(abs.(obt_D))/2

global λV = 0.0
for i in 1:α_SVD
	global λV += cartan_to_qubit_l1_treatment(CARTANS[i,:,:,:,:], true)
end
@show λT, λV, λT+λV


#=
println("\n\n\n Starting SVD routine for 1+2 body terms with cutoff tolerance $SVD_tol:")
println("Transforming spacial orbital tensors to spin-orbitals")
@time tbt_so = tbt_to_so(tbt_mo_tup, false)
@time NR_tot, SR_tot = svd_optimized(tbt_so, tol=SVD_tol, spin_orb=true)
ΔE_tot = [(SR_tot[i,2] - SR_tot[i,1])/2 for i in 1:length(NR_tot)]

println("CSA L1 bounds (NR) (SVD 1+2):")
@show sum(NR_tot)/2

println("Shifted minimal norm (SR) (SVD 1+2):")
@show sum(ΔE_tot)

println("\n\n\n Starting optimized SVD routine for 1+2 body terms with cutoff tolerance $SVD_tol:")
@time svd_optimized_so(tbt_so, tol=SVD_tol)
# =#
