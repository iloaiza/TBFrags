#CODE DRIVER FOR EXECUTION

using Distributed
@everywhere include("config.jl")

#ARGS = [1=mol_name, 2=spin_orb, 3=opt_flavour, 4=frag_flavour, 5=u_flavour, 6=α_max, 7=initial]
args_len = length(ARGS)
const mol_name = ARGS[1]

if args_len >=7
	global initial = ARGS[7]
	if initial == "false" || initial == "f"
		initial = false
	else
		initial = ARGS[7]
	end
else
	global initial = false #default no initial value loading
end

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
include("shift_grads.jl")

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
ini_cost = tbt_cost(0, tbt)
println("Initial cost is $ini_cost")

#@show Hccsd == of.hermitian_conjugated(Hccsd)S
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

if POST == true
	tbt_fin = 0 .* tbt
	for frag in FRAGS
		global tbt_fin += fragment_to_tbt(frag)
	end
	fin_cost = tbt_cost(tbt_fin, tbt)
	println("Final tbt approximated by $(round((ini_cost-fin_cost)/ini_cost*100,digits=3))%")

	psi_hf = get_wavefunction(h_ferm, "hf", num_elecs)
	psi_fci = get_wavefunction(h_ferm, "fci", num_elecs)
	SOLS = []
	println("Starting VQE optimization using mp2 decomposition fragments as generators")
	# =
	tau0 = [-0.035136392631410156, -0.005346719545802993, -0.028562273024149352, -0.0010047727141641195, -0.05231537908006339, 0.0]
	#tau0 = Float64[]
	#for nn in 1:length(FRAGS)
	for nn in 6:7
		println("Starting using $nn first fragments...")
		@time tau_sol = vqe_routine(h_ferm, psi_hf, FRAGS[1:nn],tau0=tau0)
		push!(SOLS,tau_sol)
		global tau0 = tau_sol.minimizer
	end
	println("Finished VQE optimization")
	# =#


	E_hf = expectation_value(h_ferm, psi_hf)
	E_fci = expectation_value(h_ferm, psi_fci)
	E_vqe = []
	for nn in 1:length(SOLS)
		push!(E_vqe, SOLS[nn].minimum)
	end
	@show E_hf, E_vqe, E_fci
	Ecorr = E_fci - E_hf
	@show E_fci-E_hf, -E_vqe .+ E_fci
	CORRvqe = E_vqe .- E_hf 
	ηvqe = CORRvqe ./ Ecorr

	println("Correlation energy has been approximated by:")
	@show ηvqe
end