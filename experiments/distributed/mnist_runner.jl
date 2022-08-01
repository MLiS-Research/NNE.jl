using Pkg
Pkg.activate(".")
using NNE.MNISTTraining
using TPS
using ProgressMeter
using Flux
using Dates

function clean_info_dict(dict)
    dict[:initial_state] = TPS.get_initial_state(dict[:solution].problem)
    dict[:observations] = dict[:solution].observations |> cpu
    dict[:final_state] = dict[:solution].state |> cpu
    keys_to_remove = [:solution, :labels_one_hot]
    for k in keys_to_remove
        haskey(dict, k) && delete!(dict, k)
    end
    
    all_keys = deepcopy(keys(dict))
    for key in all_keys
        if occursin("_fn", String(key))
            delete!(dict, key)
        else
            dict[key] = dict[key] |> cpu
        end
    end
    
    return dict
end

function map_params_to_trajectory(kwargs...)
    start_time = now()
    if !(typeof(kwargs) <: AbstractDict)
        kwargs = Dict(kwargs)
    end
    if haskey(kwargs, :device) && typeof(kwargs[:device]) <: Symbol
        kwargs[:device] = kwargs[:device]==:gpu ? gpu : cpu
    end

    dict = nothing
    if haskey(kwargs, :τ) && kwargs[:τ] == 1
        delete!(kwargs, :τ)
        if haskey(kwargs, :max_epochs)
            kwargs[:epochs] = kwargs[:max_epochs]
        end
        dict = clean_info_dict(solve_mnist_sa(;kwargs...))
    else
        dict = clean_info_dict(solve_mnist_trajectory(;kwargs...))
    end

    dict[:start_time] = start_time
    dict[:end_time] = now()
    return dict
end