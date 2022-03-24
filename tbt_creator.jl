#Creates tbt and saves it in .h5 file for running on clusters where tequila isn't working
#tbt used for frag_vqe.jl-like optimization, runs using gtbt.jl

#CODE DRIVER FOR EXECUTION
using Distributed
include("config.jl")

#ARGS = [1=mol_name, 2=amps_type, 3=geometry]
args_len = length(ARGS)
const mol_name = ARGS[1]
const amps_type = ARGS[2]
const geometry = parse(Float64,ARGS[3])
include("include.jl")
include("py_utils.jl")
include("shift_grads.jl")
include("tq_utils.jl")

println("Starting tequila routine with:")
@show mol_name
@show basis
@show geometry 
@show transformation
@show amps_type

Hccsd_tq, tbt_tq, molecule = tq_obtain_system(mol_name, geometry, amps_type, transformation)

tbt = tbt_tq

NAME = "TBT_"*mol_name*"_"*amps_type*"_"*string(round(geometry,digits=3))*".h5"
h5write(NAME,"tbt",tbt)
println("""Saved tbt as "$NAME".""")