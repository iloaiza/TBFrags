#CODE DRIVER FOR EXECUTION

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=mol_name, 2=save_name of initial conditions (put false if none),
#		 3=opt_flavour, 4=frag_flavour, 5=u_flavour, 6=α_max, 7=spin_orb]
args_len = length(ARGS)
const mol_name = ARGS[1]
if args_len >=2
	initial = ARGS[2]
	if initial == "false" || initial == "f"
		initial = false
	else
		initial = ARGS[2]
	end
else
	initial = false #default no initial value loading
end
if SUPPRESSOR == false
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
	if args_len >= 7
		@everywhere const spin_orb = parse(Bool, remotecall_fetch(i->ARGS[i],1,7))
	end
else
	@everywhere using Suppressor
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
	if args_len >= 7
		@everywhere @suppress_err global spin_orb = parse(Bool, remotecall_fetch(i->ARGS[i],1,7))
	end
end
@everywhere include("include.jl")

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

println("Starting calculations with:")
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

if in_block(opt_flavour)
	@show block_size
end

if singles_family(META.ff)
	global SINGLES = true
else
	global SINGLES = false
end
if include_singles == false
	if SINGLES
		error("Trying to run singles+doubles method on pure doubles tensor!")
	end
	println("Using pure two-body tensor")
	global NAME = "H_DECOMP_D_" * mol_name * "_" * frag_flavour * "_" * u_flavour * "_" * opt_flavour
	tbt, h_ferm, num_elecs = obtain_tbt(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
else
	println("Including one-body terms in two-body tensor")
	global NAME = "H_DECOMP_SD_" * mol_name * "_" * frag_flavour * "_" * u_flavour * "_" * opt_flavour

	if SINGLES == false
		if spin_orb == false
			error("Trying to run for one and two body terms with spin_orb = false, cannot combine into single two-body tensor!")
		end
		println("Combining one and two-body terms in two-body tensor")
		tbt, h_ferm, num_elecs = full_ham_tbt(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
	else
		println("Running calculations with one-body and two-body tensors tuple")
		tbt, h_ferm, num_elecs = obtain_SD(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
	end
end

FRAGS = run_optimization(opt_flavour, tbt, decomp_tol, reps, α_max, grad, verbose, x0, K0, spin_orb, NAME)

global tbt_fin = 0 .* tbt

for frag in FRAGS
    global tbt_fin += fragment_to_tbt(frag)
end


ini_cost = tbt_cost(0, tbt)
fin_cost = tbt_cost(tbt_fin, tbt)
println("Final tbt approximated by $(round((ini_cost-fin_cost)/ini_cost*100,digits=3))%")

if POST == true
	println("Starting post-decomposition routine")
	_,INIT = loading(NAME)
	x0 = INIT[1]
	K0 = INIT[2]
	H_POST(tbt, h_ferm, x0, K0, spin_orb, frag_flavour=META.ff)
end

#= Old post routine, useful for measurement problem
if POST == true
	println("Starting post-processing...")
	psi = get_wavefunction(h_ferm, wfs, num_elecs)

	num_frags = length(FRAGS)

	#LAST ENTRY HAS ONE-BODY TERM
	EXPS = zeros(num_frags + 1)
	VARS = zeros(num_frags + 1)
	COEFFS = zeros(num_frags + 1)
	h_meas = of.FermionOperator.zero()
	K0 = zeros(Int64, num_frags)

	println("Calculating expectation values and variances")
	t00 = time()
	for (i,frag) in enumerate(FRAGS)
		op = fragment_to_ferm(frag)
		norm_op = fragment_to_normalized_ferm(frag)
		global h_meas += op
		EXPS[i] = expectation_value(norm_op, psi)
		VARS[i] = variance_value(op, psi)
		COEFFS[i] = abs(frag.cn[1])
		K0[i] = frag.class
	end
	COEFFS[end] = 1
	println("Finished after $(time() - t00) seconds...")

	h_meas = h_ferm - h_meas
	println("Showing remainder term, should be one-body terms or just constant")
	@show of_simplify(h_meas)

	EXPS[end] = expectation_value(h_meas, psi)
	VARS[end] = variance_value(h_meas, psi)

	VARSUM = sum(sqrt.(VARS))
	println("Final class train:")
	@show K0
	println("Fragment coefficients: ")
	@show COEFFS
	println("Full variances:")
	@show VARS
	println("Expectations of individual reflections (from 1 to -1), last one is one-body term")
	@show EXPS .+ 1
	println("Variance metric value is $(VARSUM^2)")


	if PLOT == true
		println("Starting plotting routine")
		include("UTILS/automatic_plotting.jl")
	end

end
# =#