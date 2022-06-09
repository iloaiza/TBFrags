#  TBFrags: Two-Body Fragments package 

Factorizes two-body operators into fragments with special properties as 
H = sum_n cn Un* Sn Un^\dagger.

Fragment flavours: different classes of Sn to choose from. Unitary flavours: different kinds of Un unitaries. 


## Two-body Hamiltonian CSA decomposition
Run in bash:
'julia -p N H_DECOMP.jl mol_name'
where N is the number of processors for parallel processing, and mol_name is the name of molecule (e.g. h2).
Check run.jl and config.jl for more arguments/options
e.g.:
'julia H_DECOMP.jl lih f full-rank-non-iterative CSA MFR 8 true'
    options meaning:
- f (abbreviation for false, both can be used): doesn't load initial conditions for restarted calculation
- full-rank-non-iterative (can be written as just frni): full-rank optimization starting directly with alpha_max fragments
- CSA: type of fragment corresponding to CSA element (i.e. 2nd degree polynomial of orbitals)
- MFR: (can be written as MF-real as well) real mean-field rotations pertaining to SO(n) group
- 8: maximum number of fragments α_max. For frni optimizations it's the used number of fragments
'julia -p 5 H_DECOMP.jl lih' -> runs lih in 5 processors. Default optimization, fragment, and unitary flavours, as well as default α_max, are shown in config.jl under the names opt_flavour, frag_flavour, u_flavour, and α_max.
- true: spin-orb=true, uses spin-orbitals (false for using orbital and spin-orbital symmetry)

Most general case, we have
'julia -p N H_DECOMP.jl mol_name SAVENAME opt_flavour frag_flavour u_flavour α_max'
for N=number of OpenMP threads (i.e. shared memory parallelization)
SAVENAME explained in "SAVING RESULTS AND CALCULATION RESTARTS" last section of this file, set to false (or f) if no previous savefile.

### OPTIMIZATION FLAVOURS
Available optimizations (for obtaining decomposition) (parenthesis shows alias for quick referencing, both might be used in bash):
- greedy (g): greedy optimization, obtains each fragment as best guess of remaining operator
- relaxed-greedy (rg): greedy optimization that allows optimization of main coefficient (c_n) of previously found fragments. Uses classes of previous fragments, along with unitary rotations and any other coefficient which are not multiplicative constant c_n (not useful for CSA-type decompositions)
- full-rank (fr): performs a full-rank optimization iteratively. It first choses the first fragment in a greedy way, and uses this fragment as an initial condition for the two-fragment decomposition (while allowing the first fragment to change). This goes on, optimizing all n+1 fragments at each step while using the previously optimized n as an initial condition
- full-rank-non-iterative (frni): performs the full-rank optimization directly with all fragments. If fragment flavour has more than 1 class, the classes of the fragments are generated randomly in run.jl as "class_train" variable
- orthogonal-greedy (og): greedy optimization where each new fragment is orthogonalized with respect to all previous normalized fragments. Since finding orthogonal fragments is non-trivial, this is performed approximately by adding a penalty to the cost corresponding to the sum of inner products between the new fragment and all old ones, multiplied by a constant λort which can be set in config.jl. Choosing λort=0 means no penalty is added, but this method is still different than just greedy in this case since it optimizes the previous fragments c_n coefficients along with all of the new fragment's parameters (warning! since CSA doesn't have a c_n coefficient, it optimizes all of the previous S_n coefficients. Not debugged for CSASD or any method which includes one-body tensors)


### FRAGMENT FLAVOURS (S_n's):
- CSA: builds Cartan sub-algebra two-body polynomials for each fragment
- CSASD: CSA fragments which include one-body and two-body tensors
- CGMFR: uses all 10 classes of two-body Cartan reflections
- CRT: (Cartan Reflection Tuple) same as CGMFR, but separates one-body and two-body terms
- CR2: Same as CRT, but only includes 2-body part
- GT: (Google Tuple) same as CRT but for class k=4, corresponds to Google's unitarization ni -> 1-2ni
- G2: Same as GT, but only includes 2-body part
- GMFR: only uses 3 initially proposed classes of two-body Cartan reflections
- UPOL: builds linear combination of n1n2, (1-n1)n2, n1(1-n2), and (1-n1)(1-n2) with complex phases
- O3: builds operators with 3 eigenvalues ({-1,0,1}+const) by using p1+p2 and p1-p2 for p1, p2 in all 10 projector classes
- iCSA: like CSA, but also allows complex generators for anti-hermitian components (e.g. in1n2)
- U11: builds unitary operator from linear phases combination of projectors (11 classes, class 11 = UPOL)
- TBTON: normalized 2-body, 2-orbital polynomial based on real unitary phases (U+U') for 11th class (i.e. UPOL)
- U11R: builds unitary operator from linear phases combination of projectors with real constraint (i.e. U11+U11'). This corresponds to TBTON plus the last five reflection classes (which correspond to 3 and 4-orbital classes)
- TBPOL: two-body, two-orbital polynomial with square-root imaginary unitarization

### UNITARY FLAVOURS (U_n's):
- MF: mean-field rotations representing SU(n) group
- MF-real (MFR): real mean-field rotations, corresponding to SO(n) group


## SVD DECOMPOSITION AND FERMIONIC SYMMETRY REDUCTION
Check H_SVD.jl file for methods. Can be ran using e.g. 'julia H_SVD.jl beh2'. This file also runs all qubit-based methods (i.e. anti-commuting decomposition)
## INSTALLATION
All required packages and installation info can be seen/installed in install.sh.

Fast installation: execute install.sh in a terminal. Set PY_INSTALL=true in file to install local python environment with necessary packages. Can use custom directory for julia packages by uncommenting JL_DIR lines (useful for computing environments where writing to folder with packages is not allowed while running calculations, e.g. Niagara)
Requires installing julia packages (can be done by accessing the julia package manager in a julia session with ']', then writing: 'add Optim, PyCall, Einsum, HDF5, SharedArrays, Plots, PyPlot, Suppressor, SparseArrays, Arpack, ExpmV'). Can also be installed by running the "install.sh" script on a terminal, set PY_INSTALL to true(false) to do(not) install python environment along with julia packages. Make sure to build PyCall with correct python environment (check install.sh script, or PyCall github page for more info).

Requires/creates a python executable with installed packages:
'pip install pyscf openfermion openfermionpyscf tequila-basic h5py'

The python virtual environment should then be activated before running TBFrags routines with
'source VIRTUAL_ENVIRONMENT_DIRECTORY/bin/activate'
and making sure that julia is running with the correct JULIA_DEPOT_PATH bash variable if not using default package installation directory.


## SAVING RESULTS AND CALCULATION RESTARTS
When launching the new calculation, just run using 'julia -p N H_DECOMP.jl mol_name SAVENAME' will load the x0 and K0 values for restarting calculation with some initial conditions. Use SAVENAME=true for loading default name saved by running H_DECOMP.jl. Make sure to turn on the configuration "saving = true" to save each step of optimization for easy restarts in UTILS/config.jl.