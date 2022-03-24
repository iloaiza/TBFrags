#runs greedy optimization of tbt given in TBT savefile

#CODE DRIVER FOR EXECUTION

using Distributed
@everywhere include("UTILS/config.jl")
@everywhere include("UTILS/saving.jl")

#ARGS = [TBT_NAME, α_max]
#runs all else default
args_len = length(ARGS)
const α_max = parse(Int64, ARGS[2])
if args_len >=3
	initial = ARGS[3]
	if initial == "true" || initial == "t"
		initial = "FRAGS_"*ARGS[1]
	else
		initial = ARGS[3]
	end
else
	initial = false #default no initial value loading
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
	println("Loaded $NAMES from file $initial.h5")
else
	K0 = Int64[]
	x0 = Float64[]
end

@everywhere include("include.jl")

println("Starting tequila routine with:")
@show opt_flavour
@show frag_flavour
@show u_flavour
@show decomp_tol
@show α_max
@show grad
@show reps

global NAME = "FRAGS_"*ARGS[1]

f_name = "TBT_"*ARGS[1]*".h5"
println("""Loading tbt from file "$(f_name)".""")
tbt = h5read(f_name, "tbt")

ini_cost = tbt_cost(0, tbt)
println("Initial cost is $ini_cost")

println("Starting greedy optimization")
@time FRAGS = greedy_driver(tbt, decomp_tol, reps=reps, α_max=α_max, grad=grad, spin_orb=true, saving=true, x0=x0, K0=K0)