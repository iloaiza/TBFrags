### Most common settings for tensor representation and decomposition options
const spin_orb = false #whether spin-orbitals (true) or spacial orbitals (false) are considered. setting false changes fragment properties!
const include_singles = true #if true, include one-body term in decomposition
const decomp_tol = 2.5e-6 #decomposition tolerance for number of fragments
const α_max = 50 #maximum number of fragments
const reps = 1 #number of repetitions, useful for better exploring initial conditions space and using all processors during parallelization

### Molecular parameters/basis
const wfs = "fci" #wavefunction type for <psi|OPERATOR|psi> calculations, accepts "fci" or "hf"
const basis = "sto3g" #basis for hamiltonian
const transformation = "jw" #transformation for fermionic to qubit representation
const geometry = 1.0  #molecular geometry

const saving = true #true saves x vector and K (class-train for methods with more than one class) at each step of the optimization algorithm
######saving = false stops loading saving.jl module (no need for HDF5)
const verbose = false #set true to print current optimization values at each step, useful if calculation needs to be restarted

const grad = false #set true for using gradients during optimization, grad function needs to be implemented for flavours
const block_size = 2 #sets number of new blocks introduced at each step of block methods
const amps_type = "mp2" #default kind of two-body amplitudes for building fragments for vqe
### where normalized orbital m_i = (1/2)(n_ia + n_ib)
global NAME = "NAME" #default savename for files, usually gets changed but needs a default value

global DATAFOLDER = "SAVE/"
const NORM_ORDERED = false #whether operators are normal ordered automatically
const NORM_BRAKETS = false #whether braket operations (e.g. variances and expectations) automatically normalize operator before calculation, shouldn't change results
const GREEDY_CSA_FROM_SVD = true #set true for greedy optimization of CSA fragments to start from SVD answer, false starts from random guess

const POST = true #do post-processing once fragments are obtained (get expectation values and variances)
const PLOT = false #do correlation plot, requires POST=true

## DEFAULT FLAVOURS
const frag_flavour = "CSASD" #default fragment flavour for type of fragments built
const u_flavour = "MFR"  #default unitary flavour for type of unitaries built
const opt_flavour = "g" #default type of optimization performed

const real_tol = 1e-10 #tolerance for rounding expectation values and variances to just real component
const neg_tol = 1e-14 #tolerance for setting negative values of variance to 0, avoids complex roots
const λort = 100 #tolerance for orthogonal constraint in orthogonal-greedy algorithm, set to 0 for just adaptative greedy (adaptative part adjusts just frag coeffs corresponding to frag_num_zeros)
const SVD_tiny = 1e-8 #tolerance for checking whether SVD fragments are Hermitian or anti-Hermitian

#chose active python directory
PY_DIR = readchomp(`which python`)
if myid() == 1
	println("Using python installation in $PY_DIR")
end

## FULL-RANK OPTIMIZATION OPTIONS
const PRE_OPT = false #false for starting each new step from random x0, true for first doing greedy local optimization of one fragment (...)
					 #(...) for using inititial conditions in full-rank step

## ADDITIONAL CONFIGURATIONS
const CONFIG_LOADED = true #flag that tracks config already being loaded, useful for redefining constants and being able to load include.jl
const SUPPRESSOR = true #whether Suppressor package is used for suppresing unnecessary warnings


####################### END OF CONFIG EXECUTION OF PACKAGE LOADING
if SUPPRESSOR == true
	using Suppressor
end
