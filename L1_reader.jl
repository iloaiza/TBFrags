#SUBROUTINE FOR COLLECTING RESULTS INTO TWO-BODY TENSOR, SPIN_ORB ALWAYS TRUE
using Distributed

global mol_name = ARGS[1]

@everywhere include("UTILS/config.jl")

@everywhere @suppress_err global frag_flavour = "CSASD"
@everywhere @suppress_err global u_flavour = "MFR"

@everywhere include("include.jl")


NAME = "ILCU_SD_" * mol_name * "_MFR"
loading(NAME)
obt_orig, tbt_orig, h_ferm, num_elecs = obtain_SD(mol_name, basis=basis, ferm=true, spin_orb=true, geometry=geometry, n_elec=true)
n = size(obt_orig)[1]

n_qubit = n

fcl = frag_coeff_length(n, CSASD())
frag = fragment(x0[fcl+1:end], x0[1:fcl], 1, n_qubit, true)
obt, tbt = fragment_to_tbt(frag)

tbt_red = tbt_orig - tbt
obt_red = obt_orig - obt


H_red = tbt_to_ferm(tbt_red, true) + obt_to_ferm(obt_red, true)

tbt_full = tbt_orig + obt_to_tbt(obt_orig) #two-body tensor containing one-body and two-body terms for full Hamiltonian
tbt_red = tbt_red + obt_to_tbt(obt_red) #two-body tensor containing one-body and two-body terms for reduced (i.e. interaction picture) Hamiltonian

H_full_bk = of.bravyi_kitaev(h_ferm)
H_red_bk = of.bravyi_kitaev(H_red)

# CALCULATE AND SHOW L1 COSTS...
_, L1_ini, Pauli_cost_ini = anti_commuting_decomposition(H_full_bk)
_, L1_fin, Pauli_cost_fin = anti_commuting_decomposition(H_red_bk)

@show L1_ini, Pauli_cost_ini
@show L1_fin, Pauli_cost_fin

@show round(L1_fin / L1_ini, digits=3)
@show round(L1_ini / L1_fin, digits=3)