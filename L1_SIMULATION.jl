#Usage: run in terminal making sure that correct python virtual environment is activated as:
# julia L1_SIMULATION.jl ham_name
# with e.g. ham_name = h2
# or ham_name = h2o

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=ham_name]
args_len = length(ARGS)
const ham_name = ARGS[1]

@everywhere include("include.jl")

println("Loading calculations with:")
@show ham_name

#li and reiher load femoco Hamiltonian, and run double-factorization routine along with ancilla counting routine
#other hamiltonians run L1 routines in FULL_TREATMENT function in post_treatment.jl

if ham_name == "li" || ham_name == "reiher"
	h5name = DATAFOLDER * "eri_$(ham_name).h5"
	tbt_mo = h5read(h5name,"eri")/2 #to be consistent with 1/2 factor
	n = size(tbt_mo)[1]
	obt_mo = h5read(h5name,"h0") - sum([tbt_mo[:,k,k,:] for k in 1:n])
	tbt_mo_tup = (obt_mo, tbt_mo)
	#h_ferm = tbt_to_ferm(tbt_mo_tup, false)

	@time svd_optimized_df(tbt_mo_tup, tol=1e-6, tiny=1e-8)

	include("UTILS/ancillas.jl")
	DF_PSE_treatment(tbt_mo_tup)
else
	tbt_mo_tup, h_ferm, num_elecs = obtain_SD(ham_name, basis=basis, ferm=true, spin_orb=false, geometry=geometry, n_elec=true)
	FULL_TREATMENT(tbt_mo_tup, h_ferm, ham_name)
end

