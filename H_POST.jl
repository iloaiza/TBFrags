#LOAD AND TREAT RESULTS FROM H_DECOMP.jl
#FINDS LCU DECOMPOSITIONS OF CSASD AS REFLECTIONS
#ALSO DOES LCU USING SQRT TRICK OVER CSASD POLYNOMIALS

using Distributed
@everywhere include("UTILS/config.jl")

#ARGS = [1=mol_name, 2=opt_flavour, 3=frag_flavour, 4=u_flavour, 5=save_name of initial conditions (only necessary if different from default H_DECOMP one)]
args_len = length(ARGS)
const mol_name = ARGS[1]
if SUPPRESSOR == false
	if args_len >= 2
		@everywhere const opt_flavour = remotecall_fetch(i->ARGS[i],1,2)
	end
	if args_len >= 3
		@everywhere const frag_flavour = remotecall_fetch(i->ARGS[i],1,3)
	end
	if args_len >= 4
		@everywhere const u_flavour = remotecall_fetch(i->ARGS[i],1,4)
	end
else
	@everywhere using Suppressor
	if args_len >= 2
		@everywhere @suppress_err global opt_flavour = remotecall_fetch(i->ARGS[i],1,2)
	end
	if args_len >= 3
		@everywhere @suppress_err global frag_flavour = remotecall_fetch(i->ARGS[i],1,3)
	end
	if args_len >= 4
		@everywhere @suppress_err global u_flavour = remotecall_fetch(i->ARGS[i],1,4)
	end
end
if args_len >=5
	global NAME = ARGS[5]
else
	global NAME = false
end


@everywhere include("include.jl")

println("Loading calculations with:")
@show mol_name
@show opt_flavour
@show frag_flavour
@show u_flavour
@show basis
@show geometry 
@show spin_orb

if singles_family(META.ff)
	global SINGLES = true
else
	global SINGLES = false
end
if include_singles == false
	if SINGLES == true
		error("Trying to run singles+doubles method on pure doubles tensor!")
	end
	println("Using pure two-body tensor")
	if NAME == false
		global NAME = "H_DECOMP_D_" * mol_name * "_" * frag_flavour * "_" * u_flavour * "_" * opt_flavour
	end
	tbt, h_ferm, num_elecs = obtain_tbt(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
	n = size(tbt)[1]
else
	println("Including one-body terms in two-body tensor")
	if NAME == false
		global NAME = "H_DECOMP_SD_" * mol_name * "_" * frag_flavour * "_" * u_flavour * "_" * opt_flavour
	end

	if SINGLES == false
		if spin_orb == false
			error("Trying to run for one and two body terms with spin_orb = false, cannot combine into single two-body tensor!")
		end
		println("Combining one and two-body terms in two-body tensor")
		tbt, h_ferm, num_elecs = full_ham_tbt(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
		n = size(tbt)[1]
	else
		println("Running calculations with one-body and two-body tensors tuple")
		tbt, h_ferm, num_elecs = obtain_SD(mol_name, basis=basis, ferm=true, spin_orb=spin_orb, geometry=geometry, n_elec=true)
		n = size(tbt[1])[1]
	end
end

_,INIT = loading(NAME)
x0 = INIT[1]
K0 = INIT[2]

if spin_orb == true
	n_qubit = n
else
	n_qubit = 2n
end

H_POST(tbt, h_ferm, x0, K0, spin_orb, frag_flavour=META.ff)

#=
tbt_so = tbt_orb_to_so(tbt[2])
obt_so = obt_orb_to_so(tbt[1])
tbt_so += obt_to_tbt(obt_so)

#@show of_simplify(tbt_to_ferm(tbt, false) - tbt_to_ferm(tbt_so, true))# Sanity check, built tbt_so recovers h_ferm

#=
#L_ops, L1s = tbt_svd(tbt[2], spin_orb = false)
L_ops_so, L1s = tbt_svd(tbt_so, spin_orb=true)

num_ops = length(L_ops_so)
SVD_L1_SR = zeros(num_ops)
global TOT_OP = of.FermionOperator.zero()
for i in 1:num_ops
	println("Starting op $i")
	@time ΔE = op_range(L_ops_so[i], n_qubit)
	SVD_L1_SR[i] = (ΔE[2]-ΔE[1])/2
	global TOT_OP += L_ops_so[i]
end

@show sum(SVD_L1_SR)
@show sum(L1s)
#@show of_simplify(h_ferm - TOT_OP)

#=
#L_py = svd_py.get_one_bd_sq(tbt[2], 0)

#@show L_mats[1:3]
#@show L_py[1:3]

println("MO routine")
#@time svd_ops_MO = svd_py.get_svd_fragments(tbt[2], spin_orb=false)
@time svd_csa_MO = svd_py.get_svdcsa_sol(tbt[2])
#=
num_ops = length(svd_ops_MO)
@show num_ops
SVD_L1_SR = zeros(num_ops)

global TOT_OP = of.FermionOperator.zero()
for i in 1:num_ops
	ΔE = op_range(svd_ops_MO[i], n_qubit)
	ΔE_jl = op_range(L_ops[i], n_qubit)
	@show ΔE
	@show ΔE_jl
	SVD_L1_SR[i] = (ΔE[2]-ΔE[1])/2
	global TOT_OP += L_ops[i]
end

@show sum(SVD_L1_SR)
@show of_simplify(h_ferm - TOT_OP)


println("SO routine")
@time svd_ops = svd_py.get_svd_fragments(tbt_so, spin_orb=true)
num_ops = length(svd_ops)
@show num_ops
SVD_L1_SR = zeros(num_ops)

global TOT_OP = of.FermionOperator.zero()
for i in 1:num_ops
	ΔE = op_range(svd_ops[i], n_qubit)
	SVD_L1_SR[i] = ΔE[2]-ΔE[1]
	global TOT_OP += svd_ops[i]
end

@show sum(SVD_L1_SR)
@show of_simplify(h_ferm - TOT_OP)


#=
N_op, Sz, S2 = casimirs_builder(n_qubit)
@show of_simplify(of.commutator(N_op, h_ferm))
@show of_simplify(of.commutator(N_op, Sz))
@show of_simplify(of.commutator(N_op, S2))
# =#
# =# =# =# =# =# =#