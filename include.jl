#INCLUDE MODULE, INCLUDES EVERYTHING IN CORRECT ORDER

using LinearAlgebra, Einsum, Optim, SharedArrays

if !(@isdefined CONFIG_LOADED) #only include config file one time so constants can be later redefined
	include("config.jl")
end
include("structs.jl")
include("unitaries.jl")
include("fragments.jl")
include("cost.jl")
include("grads.jl")
include("optimization_step.jl")
include("optimization_driver.jl")

if saving == true
	include("saving.jl")
end