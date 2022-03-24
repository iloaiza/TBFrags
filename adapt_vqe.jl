#RUNS VQE USING TEQUILA, USING FROM 1 TO α_max LARGEST amps_type GENERATORS

using Distributed
@everywhere using Suppressor
@everywhere include("UTILS/config.jl")
@everywhere include("UTILS/saving.jl")

#ARGS = [1=mol_name, 2=geometry, 3=amps_type, 4=α_max, 5=transformation]

args_len = length(ARGS)
const spin_orb = true
const mol_name = ARGS[1]

if args_len >= 2
	@everywhere @suppress_err global geometry = parse(Float64,remotecall_fetch(i->ARGS[i], 1, 2))

	if args_len >= 3
	    @everywhere @suppress_err global amps_type = remotecall_fetch(i->ARGS[i],1,3)
	end
	if args_len >= 4
	    @everywhere @suppress_err global α_max = parse(Int64,remotecall_fetch(i->ARGS[i],1,4))
	end
	if args_len >= 5
	    @everywhere @suppress_err global transformation = remotecall_fetch(i->ARGS[i],1,5)
	end
end

@everywhere include("include.jl")

println("Starting tequila routine with:")
@show mol_name
@show spin_orb
@show basis
@show geometry 
@show α_max
@show transformation

global NAME = "ADAPT-VQE_" * mol_name * "_" * amps_type


ex_ops, molecule = tq_obtain_individual_ops(mol_name, geometry, α_max, amps_type=amps_type, transformation=transformation, basis=basis, spin_orb = spin_orb)
h_ferm, num_elecs = obtain_hamiltonian(mol_name, basis=basis, ferm=true, geometry=geometry, n_elec = true)


VQE_post(ex_ops, h_ferm, num_elecs, transformation=transformation, amps_type=amps_type)
