#LCU in interaction picture using greedy CSA decomp for first fragment...

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=mol_name, 2=initial]
@everywhere @suppress_err global spin_orb = true
@everywhere @suppress_err global saving = true
@everywhere @suppress_err global frag_flavour = "CSASD"
@everywhere @suppress_err global u_flavour = "MFR"
@everywhere @suppress_err global transformation = "bravyi_kitaev"
@everywhere @suppress_err global α_max = 1

@everywhere @suppress_err global reps = 5

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

@everywhere include("include.jl")
@everywhere include("UTILS/lcu_decomp.jl")

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
end

println("Starting calculations with:")
@show mol_name

global NAME = "ILCU_SD_" * mol_name * "_" * u_flavour

obt, tbt, h_ferm, num_elecs = obtain_SD(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
# =
@time FRAGS = CSASD_tot_decomp(h_ferm, reps=reps, grad=grad, verbose=verbose, spin_orb=spin_orb)

obt_fin, tbt_fin = fragment_to_tbt(FRAGS[1])

ini_cost = SD_cost(0, 0, obt, tbt)
fin_cost = SD_cost(obt_fin, tbt_fin, obt, tbt)
println("Final tbt approximated by $(round((ini_cost-fin_cost)/ini_cost*100,digits=3))%")

println("Starting anti-commuting decompositions")
# =#
H_fin = tbt_to_ferm(tbt - tbt_fin, spin_orb) + obt_to_ferm(obt - obt_fin, spin_orb)
if transformation == "bravyi_kitaev"
	H_bk_ini = of.bravyi_kitaev(h_ferm)
	H_bk = of.bravyi_kitaev(H_fin)
else
	error("Other transformation not implemented!")
end

_, L1_ini, Pauli_cost_ini = anti_commuting_decomposition(H_bk_ini)
_, L1_fin, Pauli_cost_fin = anti_commuting_decomposition(H_bk)

@show L1_ini, Pauli_cost_ini
@show L1_fin, Pauli_cost_fin

@show round(L1_fin / L1_ini, digits=3)
@show round(L1_ini / L1_fin, digits=3)