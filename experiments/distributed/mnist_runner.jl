using Pkg
Pkg.activate(".")
using NNE.MNISTTraining
using TPS
using ProgressMeter
using Flux

function dict_to_cpu(dict)
    dict[:initial_state] = TPS.get_initial_state(dict[:solution].problem)
    dict[:observations] = dict[:solution].observations |> cpu
    dict[:final_state] = dict[:solution].state |> cpu
    delete!(dict, :loss_fn)
    delete!(dict, :solution)
    delete!(dict, :model_re_fn)

    for key in keys(dict)
        dict[key] = dict[key] |> cpu
    end
    
    return dict
end

function map_params_to_trajectory(kwargs)
    if haskey(kwargs, :τ) && kwargs[:τ] == 1
        delete!(kwargs, :τ)
        return dict_to_cpu(solve_mnist_sa_automatic(;kwargs...))
    else
        return dict_to_cpu(solve_mnist_trajectory_automatic(;kwargs...))
    end

end