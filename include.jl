#INCLUDE MODULE, INCLUDES EVERYTHING IN CORRECT ORDER

using LinearAlgebra, Einsum, Optim, SharedArrays

if !(@isdefined CONFIG_LOADED) #only include config file one time so constants can be later redefined
	include("UTILS/config.jl")
end

if SUPPRESSOR == true
	using Suppressor
end

if !(@isdefined SAVING_LOADED) && saving == true #only include saving file one time if saving option is on
	include("UTILS/saving.jl")
	global SAVING_LOADED = true
end

include("UTILS/structs.jl")
include("UTILS/unitaries.jl")
include("UTILS/fragments.jl")
include("UTILS/tensor_utils.jl")
include("UTILS/svd.jl")
include("UTILS/cost.jl")
include("UTILS/grads.jl")
include("UTILS/optimization_step.jl")
include("UTILS/optimization_driver.jl")
include("UTILS/py_utils.jl")
include("UTILS/casimir_reduction.jl")
include("UTILS/shift_grads.jl")
include("UTILS/tq_utils.jl")
include("UTILS/pauli_utils.jl")
include("UTILS/post_treatment.jl")
