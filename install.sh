#!/bin/bash
# creates and downloads python virtual environment with necessary packages if PY_INSTALL = "true", 0 skips to julia instalation
# installs all necessary julia packages for running. Requires an existing installation of both julia and python
# run script while python environment is active
PY_INSTALL=true 

if $PY_INSTALL
then
	echo 'Creating python environment and installing python packages...'
	#loads module, for environments where necessary for running python...
	module load python/3.7
	#--system-site-packages option makes sure configuration from system is taken for packages (e.g. OpenBLAS)
	virtualenv --system-site-packages venv
	source venv/bin/activate
	pip install openfermion pyscf openfermionpyscf tequila-basic
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
ENV["PYTHON"] = Sys.which("python")
println("""Building PyCall with python executable $(ENV["PYTHON"])""")
Pkg.build("PyCall")' > install.jl
echo 'Starting julia packages installation...'
julia install.jl
echo 'Finished packages installation (unless julia package manager error message)'
rm install.jl