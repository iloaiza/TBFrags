#LOAD AND TREAT RESULTS FROM H_DECOMP.jl
#FINDS LCU DECOMPOSITIONS OF CSASD AS REFLECTIONS
#ALSO DOES LCU USING SQRT TRICK OVER CSASD POLYNOMIALS

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=mol_name]
args_len = length(ARGS)
const mol_name = ARGS[1]


@everywhere include("include.jl")

println("Loading calculations with:")
@show mol_name
@show basis
@show geometry 

tbt, h_ferm, num_elecs = full_ham_tbt(mol_name, basis=basis, ferm=true, spin_orb=true, geometry=geometry, n_elec=true)

n = size(tbt)[1]
n_qubit = n

H_TREATMENT(tbt, h_ferm, true, S2=true)
