using NNE.MNISTTraining
using TPS
using Flux
using Dates
using Random
using UUIDs
using Distributed
import Distributed: myid
using CUDA

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

function map_params_to_trajectory(kwargs, trial_id::UUID)
    start_time = now()
    if !(typeof(kwargs) <: Dict)
        kwargs = Dict(kwargs)
    end

    if haskey(kwargs, :device) && typeof(kwargs[:device]) <: Symbol
        kwargs[:device] = kwargs[:device] == :gpu ? gpu : cpu
    end

    dict = nothing
    if haskey(kwargs, :seed)
        Random.seed!(kwargs[:seed])
    end
    if haskey(kwargs, :τ) && kwargs[:τ] == 1
        delete!(kwargs, :τ)
        if haskey(kwargs, :max_epochs)
            kwargs[:epochs] = kwargs[:max_epochs]
        end
        dict = clean_info_dict(solve_mnist_sa(; trial_id=trial_id, kwargs...))
    else
        dict = clean_info_dict(solve_mnist_trajectory(; trial_id=trial_id, kwargs...))
    end

    dict[:start_time] = start_time
    dict[:end_time] = now()
    dict[:duration] = dict[:end_time] - dict[:start_time]
    dict[:git_hash] = get_git_hash()
    dict[:worker_id] = Distributed.myid()
    CUDA.reclaim()
    return dict
end

function get_git_hash()
    return strip(read(`git rev-parse HEAD`, String))
end