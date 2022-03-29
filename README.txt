#############################################################################
#################### TBFrags: Two-Body Fragments package ####################
#############################################################################


Factorizes two-body operators into fragments with special properties as H = sum_n c_n U_n^* S_n U_n (e.g. S_n^2 = 1).


######## HOW TO USE:
## Interaction picture simulations:
Run in bash: (e.g. for lih molecule)
'julia ILCU.jl lih'
Does greedy CSA decomposition simultaneously of one-body and two-body terms for 1 fragment to build interaction picture.
Then calculates L1 norms with naive Pauli, and two different flavours of anticommuting groupings


## Two-body Hamiltonian decomposition
Run in bash:
'julia -p N run.jl mol_name'
where N is the number of processors for parallel processing, and mol_name is the name of molecule (e.g. h2).
Check run.jl and config.jl for more arguments/options
e.g.:
'julia run.jl lih f full-rank-non-iterative CSA MFR 8'
    options meaning:
        - f (abbreviation for false, both can be used): doesn't load initial conditions for restarted calculation
        - full-rank-non-iterative (can be written as just frni): full-rank optimization starting directly with alpha_max fragments
        - CSA: type of fragment corresponding to CSA element (i.e. 2nd degree polynomial of orbitals)
        - MFR: (can be written as MF-real as well) real mean-field rotations pertaining to SO(n) group
        - 8: maximum number of fragments α_max. For frni optimizations it's the used number of fragments
'julia -p 5 run.jl lih' -> runs lih in 5 processors. Default optimization, fragment, and unitary flavours, as well as default α_max, are shown in config.jl under the names opt_flavour, frag_flavour, u_flavour, and α_max.

Most general case, we have
'julia -p N run.jl mol_name SAVENAME opt_flavour frag_flavour u_flavour α_max'
SAVENAME explained in "SAVING RESULTS AND CALCULATION RESTARTS" last section of this file.

## VQE routines: check vqe drivers...

######## IMPLEMENTED FLAVOURS
Available optimizations (for obtaining decomposition) (parenthesis shows alias for quick referencing):
    - greedy (g): greedy optimization, obtains each fragment as best guess of remaining operator

    - relaxed-greedy (rg): greedy optimization that allows optimization of main coefficient (c_n) of previously found fragments. Uses classes of previous fragments, along with unitary rotations and any other coefficient which are not multiplicative constant c_n (not useful for CSA-type decompositions)

    - full-rank (fr): performs a full-rank optimization iteratively. It first choses the first fragment in a greedy way, and uses this fragment as an initial condition for the two-fragment decomposition (while allowing the first fragment to change). This goes on, optimizing all n+1 fragments at each step while using the previously optimized n as an initial condition

    - full-rank-non-iterative (frni): performs the full-rank optimization directly with all fragments. If fragment flavour has more than 1 class, the classes of the fragments are generated randomly in run.jl as "class_train" variable

    - orthogonal-greedy (og): greedy optimization where each new fragment is orthogonalized with respect to all previous normalized fragments. Since finding orthogonal fragments is non-trivial, this is performed approximately by adding a penalty to the cost corresponding to the sum of inner products between the new fragment and all old ones, multiplied by a constant λort which can be set in config.jl. Choosing λort=0 means no penalty is added, but this method is still different than just greedy in this case since it optimizes the previous fragments c_n coefficients along with all of the new fragment's parameters
    (warning! since CSA doesn't have a c_n coefficient, it optimizes all of the previous S_n coefficients)


Available fragments (S_n's):
    - CGMFR: uses all 10 classes of two-body Cartan reflections

    - GMFR: only uses 3 initially proposed classes of two-body Cartan reflections

    - UPOL: builds linear combination of n1n2, (1-n1)n2, n1(1-n2), and (1-n1)(1-n2) with complex phases

    - O3: builds operators with 3 eigenvalues ({-1,0,1}+const) by using p1+p2 and p1-p2 for p1, p2 in all 10 projector classes

    - CSA: builds Cartan sub-algebra two-body polynomials for each fragment

    - iCSA: like CSA, but also allows complex generators for anti-hermitian components (e.g. in1n2)

    - U11: builds unitary operator from linear phases combination of projectors (11 classes, class 11 = UPOL)

    - TBTON: normalized 2-body, 2-orbital polynomial based on real unitary phases (U+U') for 11th class (i.e. UPOL)

    - U11R: builds unitary operator from linear phases combination of projectors with real constraint (i.e. U11+U11'). This corresponds to TBTON plus the last five reflection classes (which correspond to 3 and 4-orbital classes)


Available unitary rotations (U_n's):
    - MF: mean-field rotations representing SU(n) group

    - MF-real (MFR): real mean-field rotations, corresponding to SO(n) group


######## REQUIRED PACKAGES:
julia:
    -Optim, PyCall, Einsum, SharedArrays
    -HDF5 (for saving)
    -Plots, PyPlot (for plottling)
    -Suppressor (for suppressing unnecessary warnings)
    -SparseArrays, Arpack and ExpmV: for efficient calculation and storage of wavefunctions and operators, only used for openfermion based vqe routine (shift_grads.jl)
python:
    -openfermion, pyscf, openfermionpyscf, tequila-basic


######## INSTALLATION
Requires installing julia packages (can be done by accessing the julia package manager in a julia session with ']', then writing: 'add Optim, PyCall, Einsum, HDF5, SharedArrays, Plots, PyPlot, Suppressor, SparseArrays, Arpack, ExpmV'). Can also be installed by running the "install.sh" script on a terminal.
Also requires a python executable with installed packages:
'pip install pyscf openfermion openfermionpyscf'

If running in a server, a python environment with openfermion can be installed with: (will create "venv" folder with installation in current directory)
'''
module load python/3.7
virtualenv --system-site-packages venv
source venv/bin/activate
pip install openfermion pyscf openfermionpyscf tequila-basic
'''
(name of python module should be changed depending on server module names)

The python virtual environment should then be activated before running run.jl by the command
'source venv/bin/activate'


######## SAVING RESULTS AND CALCULATION RESTARTS
When launching the new calculation, just run using 'julia -p N run.jl mol_name SAVENAME' will load the x0 and K0 values for restarting calculation with some initial conditions. Use option "saving = true" to save each step of optimization for easy restarts