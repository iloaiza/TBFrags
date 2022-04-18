#LOAD AND TREAT RESULTS FROM H_DECOMP.jl
#FINDS LCU DECOMPOSITIONS OF CSASD AS REFLECTIONS
#ALSO DOES LCU USING SQRT TRICK OVER CSASD POLYNOMIALS

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=mol_name, 2=spin_orb]
args_len = length(ARGS)
const mol_name = ARGS[1]
if length(ARGS) >= 2
	@everywhere @suppress_err global spin_orb = parse(Bool, remotecall_fetch(i->ARGS[i],1,2))
end

@everywhere include("include.jl")

println("Loading calculations with:")
@show mol_name
@show basis
@show geometry 
@show spin_orb

tbt, h_ferm, num_elecs = obtain_SD(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
#if you only have tbt, you can get h_ferm by doing h_ferm = tbt_to_ferm(tbt, spin_orb)

tbt_so = tbt_to_so(tbt, spin_orb)
SVD_CARTAN_TBTS, SVD_TBTS = tbt_svd(tbt_so, tol=1e-6, spin_orb=true)

α_SVD = size(SVD_TBTS)[1]
println("SVD sanity (difference should be small):")
svd_tbt = SVD_TBTS[1,:,:,:,:]
for i in 2:α_SVD
	svd_tbt += SVD_TBTS[i,:,:,:,:]
end
@show sum(abs.(tbt_so - svd_tbt))

n = size(tbt_so)[1]
n_qubit = n
H_full_q = qubit_transform(h_ferm)
qubit_treatment(H_full_q)