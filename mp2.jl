#CODE DRIVER FOR EXECUTION

using Distributed
@everywhere include("config.jl")

#ARGS = [1=mol_name, 2=spin_orb, 3=opt_flavour, 4=frag_flavour, 5=u_flavour, 6=α_max]
args_len = length(ARGS)
const mol_name = ARGS[1]
initial = false

if SUPPRESSOR == false
	if args_len >= 2
		@everywhere const spin_orb = parse(Bool, remotecall_fetch(i->ARGS[i],1,2))
	end
	if args_len >= 3
		@everywhere const opt_flavour = remotecall_fetch(i->ARGS[i],1,3)
	end
	if args_len >= 4
		@everywhere const frag_flavour = remotecall_fetch(i->ARGS[i],1,4)
	end
	if args_len >= 5
		@everywhere const u_flavour = remotecall_fetch(i->ARGS[i],1,5)
	end
	if args_len >= 6
		@everywhere const α_max = parse(Int64, remotecall_fetch(i->ARGS[i],1,6))
	end
else
	@everywhere using Suppressor
	if args_len >= 2
		@everywhere @suppress_err global spin_orb = parse(Bool, remotecall_fetch(i->ARGS[i],1,2))
	end
	if args_len >= 3
		@everywhere @suppress_err global opt_flavour = remotecall_fetch(i->ARGS[i],1,3)
	end
	if args_len >= 4
		@everywhere @suppress_err global frag_flavour = remotecall_fetch(i->ARGS[i],1,4)
	end
	if args_len >= 5
		@everywhere @suppress_err global u_flavour = remotecall_fetch(i->ARGS[i],1,5)
	end
	if args_len >= 6
		@everywhere @suppress_err global α_max = parse(Int64, remotecall_fetch(i->ARGS[i],1,6))
	end
end
@everywhere include("include.jl")
include("py_utils.jl")

if initial != false
	NAMES,INIT = loading(initial)
	if NAMES[1] == "x0"
		x0 = INIT[1]
		K0 = INIT[2]
	else
		x0 = INIT[2]
		K0 = INIT[1]
	end
	println("Loaded $NAMES")
else
	K0 = Int64[]
	x0 = Float64[]
end

println("Starting MP2 calculations with:")
@show mol_name
@show opt_flavour
@show frag_flavour
@show u_flavour
@show spin_orb
@show wfs
@show basis
@show geometry 
@show decomp_tol
@show α_max
@show verbose
@show grad
@show reps

h_ferm, num_elecs = obtain_hamiltonian(mol_name, basis=basis, ferm=true, geometry=geometry, n_elec = true)
tbt, Hccsd = obtain_ccsd(mol_name, basis=basis, spin_orb=spin_orb, geometry=geometry)

#=
E = of.eigenspectrum(Hccsd)

@show E
@show length(E)
@show array_rounder(E,10)
@show length(array_rounder(E,10))
# =#

num_orbs = length(tbt[:,1,1,1])
println("Using two-body tensor representation with $num_orbs orbitals")
# =
if opt_flavour == "full-rank" || opt_flavour == "fr"
	@time FRAGS = full_rank_driver(tbt, decomp_tol, reps = reps, α_max=α_max, grad=grad, verbose=verbose, x0=x0, K0=K0, spin_orb=spin_orb)
elseif opt_flavour == "greedy" || opt_flavour == "g"
	@time FRAGS = greedy_driver(tbt, decomp_tol, reps = reps, α_max=α_max, grad=grad, verbose=verbose, spin_orb=spin_orb)
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

if POST == true
	println("Starting post-processing...")
	psi = get_wavefunction(h_ferm, wfs)

	num_frags = length(FRAGS)

	#LAST ENTRY HAS ONE-BODY TERM
	EXPS = zeros(num_frags + 1)
	VARS = zeros(num_frags + 1)
	COEFFS = zeros(num_frags + 1)
	h_meas = of.FermionOperator.zero()

	println("Calculating expectation values and variances")
	t00 = time()
	for (i,frag) in enumerate(FRAGS)
		op = fragment_to_ferm(frag)
		global h_meas += op
		EXPS[i] = expectation_value(op, psi)
		VARS[i] = variance_value(op, psi)
		COEFFS[i] = abs(frag.cn[1])
	end
	COEFFS[end] = 1
	println("Finished after $(time() - t00) seconds...")

	h_meas = Hccsd - h_meas
	#@show of.normal_ordered(h_meas)

	EXPS[end] = expectation_value(h_meas, psi)
	VARS[end] = variance_value(h_meas, psi)

	VARSUM = sum(sqrt.(VARS))
	println("Fragment coefficients: ")
	@show COEFFS
	println("Full variances:")
	@show VARS
	println("Expectations of individual reflections (from 1 to -1), last one is one-body term")
	@show EXPS .+ 1
	println("Variance metric value is $(VARSUM^2)")


	if PLOT == true
		println("Starting plotting routine")
		include("automatic_plotting.jl")
	end

end