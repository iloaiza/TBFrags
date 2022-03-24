#CODE DRIVER FOR EXECUTION

using Distributed
@everywhere using Suppressor
@everywhere include("UTILS/config.jl")

#ARGS = [1=mol_name, 2=α_max, 3=u_flavour, 4=amps_type]
args_len = length(ARGS)
const mol_name = ARGS[1]
const spin_orb = true
@everywhere @suppress_err global frag_flavour = "CCD"

if args_len >= 2
    @everywhere @suppress_err global α_max = parse(Int64, remotecall_fetch(i->ARGS[i],1,2))
end
if args_len >= 3
    @everywhere @suppress_err global u_flavour = remotecall_fetch(i->ARGS[i],1,3)
end
if args_len >= 4
    @everywhere @suppress_err global amps_type = remotecall_fetch(i->ARGS[i],1,4)
end

@everywhere include("include.jl")

println("Starting tequila routine with:")
@show mol_name
@show frag_flavour
@show u_flavour
@show spin_orb
@show basis
@show geometry 
@show α_max
@show transformation


Hccsd_tq, tbt_tq, molecule, tbt_herm = tq_obtain_direct_ccop(mol_name, geometry, amps_type, transformation)
h_ferm, num_elecs = obtain_hamiltonian(mol_name, basis=basis, ferm=true, geometry=geometry, n_elec = true)

println("Starting MP2 operator decomposition...")
@time FRAGS = greedy_ccd_mf(tbt_tq, α_max)


global tbt_fin = 0 .* tbt_herm
for frag in FRAGS
    global tbt_fin += fragment_to_tbt(frag)
end

ini_cost = tbt_cost(0, tbt_herm)
fin_cost = tbt_cost(tbt_fin, tbt_herm)
println("Final tbt approximated by $(round((ini_cost-fin_cost)/ini_cost*100,digits=3))%")


if POST == true
    VQE_post(FRAGS, h_ferm, num_elecs, transformation=transformation, amps_type=amps_type)
end