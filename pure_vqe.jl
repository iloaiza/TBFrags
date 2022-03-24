#RUNS VQE ON PRECALCULATED FRAGMENTS, ENTER FILENAME IN ARGS[2] (without .h5)

using Distributed
using Suppressor
@everywhere include("UTILS/config.jl")
@everywhere include("UTILS/saving.jl")

#ARGS = [1=initial, 2=mol_name, 3=geometry, 4=amps_type, 5=α_max, 6=transformation]

args_len = length(ARGS)
const spin_orb = true
const initial = ARGS[1]

global α_set = false
if args_len >= 2
	const mol_name = ARGS[2]

	if args_len >= 3
    @everywhere @suppress_err global geometry = parse(Float64, remotecall_fetch(i->ARGS[i],1,3))
	end
	if args_len >= 4
	    @everywhere @suppress_err global amps_type = remotecall_fetch(i->ARGS[i],1,4)
	end
	if args_len >= 5
	    @everywhere @suppress_err global α_max = parse(Int64,remotecall_fetch(i->ARGS[i],1,5))
	    global α_set = true
	end
	if args_len >= 6
	    @everywhere @suppress_err global transformation = remotecall_fetch(i->ARGS[i],1,6)
	end
else
	println("Obtaining molecule info from file")
	S_list = split(initial,"_")
	if S_list[1] == "FRAGS"
		i_start = 2
	else
		i_start = 1
	end
	const mol_name = S_list[i_start]
	if amps_type != S_list[i_start+1]
		@everywhere @suppress_err global amps_type = String(S_list[i_start+1])
	end
	if geometry != S_list[i_start+2]
		@everywhere @suppress_err global geometry = parse(Float64,S_list[i_start+2])
	end
end

@everywhere include("include.jl")

NAMES,INIT = loading(initial)
if NAMES[1] == "x0"
    x0 = INIT[1]
    K0 = INIT[2]
else
    x0 = INIT[2]
    K0 = INIT[1]
end
println("Loaded $NAMES from $initial.h5")
#@show x0
#@show K0

if α_set == false
	println("No α_max in input, using all saved values...")
	@everywhere @suppress_err global α_max = length(K0)
	println("α_max = $α_max")
end

println("Starting tequila routine with:")
@show mol_name
@show opt_flavour
@show frag_flavour
@show u_flavour
@show spin_orb
@show basis
@show geometry 
@show decomp_tol
@show α_max
@show verbose
@show grad
@show reps
@show transformation

global NAME = mol_name * "_" * amps_type * "_" * frag_flavour * "_" * u_flavour * "_" * opt_flavour

Hccsd_tq, tbt_tq, molecule = tq_obtain_system(mol_name, geometry, amps_type, transformation)
h_ferm, num_elecs = obtain_hamiltonian(mol_name, basis=basis, ferm=true, geometry=geometry, n_elec = true)

global tbt = tbt_tq
ini_cost = tbt_cost(0, tbt)
println("Initial cost is $ini_cost")

num_orbs = length(tbt[:,1,1,1])
println("Using two-body tensor representation with $num_orbs orbitals")

# =
if opt_flavour == "full-rank" || opt_flavour == "fr"
	@time FRAGS = full_rank_driver(tbt, decomp_tol, reps = reps, α_max=α_max, grad=grad, verbose=verbose, x0=x0, K0=K0, spin_orb=spin_orb)
elseif opt_flavour == "greedy" || opt_flavour == "g"
	@time FRAGS = greedy_driver(tbt, decomp_tol, reps = reps, α_max=α_max, grad=grad, verbose=verbose, spin_orb=spin_orb, x0=x0, K0=K0)
elseif opt_flavour == "og" || opt_flavour == "orthogonal-greedy"
	println("Using λ=$λort for orthogonal greedy constrain value")
	@time FRAGS = orthogonal_greedy_driver(tbt, decomp_tol, reps = reps, α_max=α_max, grad=grad, verbose=verbose, spin_orb=spin_orb)
elseif opt_flavour == "frni" || opt_flavour == "full-rank-non-orthogonal"
	num_classes = number_of_classes(frag_flavour)
	class_train = rand(1:num_classes, α_max)
	@show class_train
	@time FRAGS = full_rank_non_iterative_driver(tbt, grad=grad, verbose=verbose, x0=x0, K0=class_train, spin_orb=spin_orb)
else
	error("Trying to do decomposition with optimization flavour $opt_flavour, not implemented!")
end

global tbt_fin = 0 .* tbt
for frag in FRAGS
	global tbt_fin += fragment_to_tbt(frag)
end
fin_cost = tbt_cost(tbt_fin, tbt)
println("Final tbt approximated by $(round((ini_cost-fin_cost)/ini_cost*100,digits=3))%")


# VQE post-processing

if POST == true
	VQE_post(FRAGS, h_ferm, num_elecs, transformation=transformation, amps_type=amps_type)
end