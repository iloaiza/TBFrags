#!/bin/bash
# creates and downloads python virtual environment with necessary packages if PY_INSTALL = "true", 0 skips to julia instalation
# installs all necessary julia packages for running. Requires an existing installation of both julia and python
# run script while python environment is active

# true creates a python environment in a folder
PY_INSTALL=false 
# julia directory for packages, uncomment both lines for installing julia in a particular folder
#JL_DIR="../../JULIA/.julia"
#export JULIA_DEPOT_PATH=$JL_DIR


if $PY_INSTALL
then
	echo 'Creating python environment and installing python packages...'
	#loads module, for environments where necessary for running python...
	module load python/3.7
	#--system-site-packages option makes sure configuration from system is taken for packages (e.g. OpenBLAS)
	virtualenv --system-site-packages py_env
	source ./py_env/bin/activate
	pip install openfermion sympy pyscf openfermionpyscf tequila-basic h5py
fi

touch install.jl
echo 'import Pkg
Pkg.add("Optim")
Pkg.add("PyCall")
Pkg.add("Einsum")
Pkg.add("HDF5")
Pkg.add("SharedArrays")
Pkg.add("Suppressor")
Pkg.add("Plots")
Pkg.add("SparseArrays")
Pkg.add("Arpack")
Pkg.add("ExpmV")
Pkg.add("PyPlot")
Pkg.add("HiGHS")
Pkg.add("JuMP")
ENV["PYTHON"] = Sys.which("python")
ENV["JL_RUNTIME_PYTHON"] = Sys.which("python")
println("""Building PyCall with python executable $(ENV["PYTHON"])""")
Pkg.build("PyCall")
using PyCall' > install.jl
echo 'Starting julia packages installation...'
julia install.jl
echo 'Finished packages installation (unless julia package manager error message)'
rm install.jl