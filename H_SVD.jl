#LOAD AND TREAT RESULTS FROM H_DECOMP.jl
#FINDS LCU DECOMPOSITIONS OF CSASD AS REFLECTIONS
#ALSO DOES LCU USING SQRT TRICK OVER CSASD POLYNOMIALS

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=mol_name, 2=spin_orb]
args_len = length(ARGS)
const mol_name = ARGS[1]
if length(ARGS) >= 2
	@everywhere @suppress_err global spin_orb = parse(Bool, remotecall_fetch(i->ARGS[i],1,2))
end

@everywhere include("include.jl")

println("Loading calculations with:")
@show mol_name
@show basis
@show geometry 
@show spin_orb

#tbt, h_ferm, num_elecs = full_ham_tbt(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
tbt, h_ferm, num_elecs = obtain_SD(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)

S2 = false
@show S2
#H_TREATMENT(tbt, h_ferm, spin_orb, S2=S2)
QUBIT_TREATMENT(tbt, h_ferm, spin_orb, S2=S2)
