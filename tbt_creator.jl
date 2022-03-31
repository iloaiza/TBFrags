#Creates tbt and saves it in .h5 file for running on clusters where tequila isn't working
#tbt used for frag_vqe.jl-like optimization, runs using gtbt.jl

#CODE DRIVER FOR EXECUTION
using Distributed
include("UTILS/config.jl")

#ARGS = [1=mol_name]

const mol_name = ARGS[1]
include("include.jl")

println("Starting tequila routine with:")
@show mol_name
@show basis
@show geometry 

if include_singles == true
	println("Creating tbt with one and two-body terms")
	NAME = "TBT_SD_"*mol_name*".h5"
	tbt, h_ferm, num_elecs = full_ham_tbt(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
else
	println("Creating tbt with only two-body terms")
	NAME = "TBT_D_"*mol_name*".h5"
	tbt, h_ferm, num_elecs = obtain_tbt(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
end

h5write(DATAFOLDER*NAME,"tbt",tbt)
println("""Saved tbt as "$NAME".""")