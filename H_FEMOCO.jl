#LOAD AND TREAT RESULTS FROM H_DECOMP.jl
#FINDS LCU DECOMPOSITIONS OF CSASD AS REFLECTIONS
#ALSO DOES LCU USING SQRT TRICK OVER CSASD POLYNOMIALS

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=ham_name]
args_len = length(ARGS)
const ham_name = ARGS[1]

@everywhere include("include.jl")

println("Loading calculations with:")
@show ham_name

tbt_so, h_ferm = load_full_ham_tbt(ham_name, spin_orb=false, prefix=".")

tol = 1e-8
println("Starting SVD routine with cutoff tolerance $tol:")
@time SVD_CARTAN_TBTS, SVD_TBTS = tbt_svd(tbt_so, tol=tol, spin_orb=true)

n = size(tbt_so)[1]
n_qubit = n
println("Performing fermion to qubit mapping:")
@time H_full_q = qubit_transform(h_ferm)

println("Qubit treatment of Hamiltonian:")
qubit_treatment(H_full_q)
