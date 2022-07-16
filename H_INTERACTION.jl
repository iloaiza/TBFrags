#CODE DRIVER FOR EXECUTION

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=mol_name, 2=save_name of initial conditions (put false or f if none, true or t for default name from last run),
#		 3=opt_flavour, 4=frag_flavour, 5=u_flavour, 6=α_max, 7=spin_orb]
args_len = length(ARGS)
const mol_name = ARGS[1]
if args_len >=2
	initial = ARGS[2]
	if initial == "false" || initial == "f"
		initial = false
	elseif initial == "true" || initial == "t"
		initial = true
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


global NAME = "H_INTERACTION_SVD_" * mol_name * "_" * frag_flavour * "_" * u_flavour
if initial == true
	initial = NAME
end

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

if SINGLES == false
	n = size(tbt)[1]
else
	n = size(tbt[1])[1]
end


# =
tot_cost = tbt_cost(0, tbt)
FRAG_INTER = greedy_driver(tbt, 0.99*tot_cost, reps = reps, α_max=1,
						   grad=grad, verbose=verbose, spin_orb=spin_orb,
						   x0=x0, K0=K0, f_name=NAME)
tbt_inter = fragment_to_tbt(FRAG_INTER[1], frag_flavour=CSASD(), u_flavour=MF_real())
tbt_targ = tbt - tbt_inter
inter_cost = tbt_cost(0, tbt_targ)
println("Interaction picture main component CSA tbt approximates total by $(round((tot_cost-inter_cost)/tot_cost*100,digits=3))%")
rem_op_CSA = of_simplify(h_ferm - tbt_to_ferm(tbt_inter, spin_orb))
RANGE = op_range(rem_op_CSA)
ΔE = (RANGE[2] - RANGE[1])/2
@show ΔE

#SHIFT_TREATMENT(tbt_targ, rem_op_CSA, mol_name)
FULL_TREATMENT(tbt_targ, rem_op_CSA, "INTERACTION_CSA_"*mol_name)
# =#

#=
tbt_so = tbt_to_so(tbt, spin_orb)
tot_cost = tbt_cost(0, tbt_so)
cns, u_params = tbt_svd_1st(tbt_so)
frag_svd = fragment(u_params, cns, 1, 2n, true)
tbt_inter = fragment_to_tbt(frag_svd, frag_flavour=CSA(), u_flavour=MF_real())
tbt_targ = tbt_so - tbt_inter
inter_cost = tbt_cost(0, tbt_targ)
println("Interaction picture main component SVD tbt approximates total by $(round((tot_cost-inter_cost)/tot_cost*100,digits=3))%")

rem_op_SVD = of_simplify(h_ferm - tbt_to_ferm(tbt_inter, true))
RANGE = op_range(rem_op_SVD)
ΔE = (RANGE[2] - RANGE[1])/2
@show ΔE

FULL_TREATMENT(tbt_targ, rem_op_SVD, "INTERACTION_SVD_"*mol_name)
# =#