#LOAD AND TREAT RESULTS FROM H_DECOMP.jl
#FINDS LCU DECOMPOSITIONS OF CSASD AS REFLECTIONS
#ALSO DOES LCU USING SQRT TRICK OVER CSASD POLYNOMIALS

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=ham_name]
args_len = length(ARGS)
const ham_name = ARGS[1]

@everywhere include("include.jl")

println("Loading calculations with:")
@show ham_name

tbt_so, h_ferm = load_full_ham_tbt(ham_name, spin_orb=false, prefix=".")

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
