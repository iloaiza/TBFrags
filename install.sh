# installs all necessary julia packages for running. Requires an existing installation of both julia and python

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
Pkg.add("PyPlot")' > install.jl
echo 'Starting julia packages installation...'
julia install.jl
echo 'Finished packages installation (unless julia package manager error message)'
rm install.jl