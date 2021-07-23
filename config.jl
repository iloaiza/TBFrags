const wfs = "fci" #wavefunction type for <psi|OPERATOR|psi> calculations
const basis = "sto3g" #basis for hamiltonian
const geometry = 1.0  #molecular geometry
const decomp_tol = 2.5e-6 #decomposition tolerance for number of fragments
const α_max = 500 #maximum number of fragments
const verbose = true #set true to print current optimization values at each step, useful if calculation needs to be restarted
const saving = false #true to save x vector and K (class-train for methods with more than one class) at each step of the optimization algorithm
######saving = false stops loading saving.jl module (no need for HDF5)
const grad = false #set true for using gradients during optimization, grad function needs to be implemented for flavours
const reps = 1 #number of repetitions, useful for better exploring initial conditions space and using all processors during parallelization
const spin_orb = false #whether spin-orbitals (true) or orbitals (false) are considered. setting false changes fragment properties!

const NORM_ORDERED = false #whether operators are normal ordered automatically

const POST = true #do post-processing once fragments are obtained (get expectation values and variances)
const PLOT = true #do correlation plot, requires POST=true

## DEFAULT FLAVOURS
const frag_flavour = "CGMFR" #default fragment flavour for type of fragments built
const u_flavour = "MF-real"  #default unitary flavour for type of unitaries built
const opt_flavour = "full-rank" #default type of optimization performed

const real_tol = 1e-10 #tolerance for rounding expectation values and variances to just real component
const neg_tol = 1e-14 #tolerance for setting negative values of variance to 0, avoids complex roots
const λort = 100 #tolerance for orthogonal constraint in orthogonal-greedy algorithm, set to 0 for just adaptative greedy (adaptative part adjusts just frag coeffs corresponding to frag_num_zeros)

#chose active python directory
PY_DIR = readchomp(`which python`)
println("Using python installation in $PY_DIR")

# FULL-RANK OPTIMIZATION OPTIONS
const PRE_OPT = false #false for starting each new step from random x0, true for first doing greedy local optimization of one fragment (...)
					 #(...) for using inititial conditions in full-rank step


const CONFIG_LOADED = true
const SUPPRESSOR = true #whether Suppressor package is used for suppresing unnecessary warnings