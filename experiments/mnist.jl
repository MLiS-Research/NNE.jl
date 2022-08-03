using Distributed
@everywhere include("distributed/mnist_runner.jl")
include("run_helper.jl")
import Base.Iterators: product
using BSON: @save, @load
using ProgressMeter
using Statistics
using Flux
using CUDA
using TPS
using NNE
using NNE.Runner
using TPS.Convergence

function get_experiment_parameter_dictionaries(;device=:gpu, epochs=1_000_000)
    s_min = 0.01
    s_max = 5.0
    num_s_values = 9
    s_values = get_exponentially_spaced(s_min, s_max, num_s_values)
    trajectory_lengths = [1, 2, 4, 8, 16]
    options = Dict{Symbol, Any}()
    options[:fraction_to_include] = 0.25
    options[:epochs] = epochs
    options[:device] = device
    options[:n_samples] = 2048
    options[:outputs] = 2
    options[:σ] = 0.05
    function construct_dict(s, τ)
        new_options = deepcopy(options)
        new_options[:s] = s
        new_options[:τ] = τ
        return new_options
    end
    input_dictionaries = [construct_dict(s, τ) for (s, τ) in product(s_values, trajectory_lengths)]
    return input_dictionaries
end

function mnist_trajectory_experiment(; mode::TaskExecutionMode=SerialMode, device=:gpu, show_progress=false, kwargs...)
    input_dictionaries = get_experiment_parameter_dictionaries(; device=device, kwargs...)
    map_params_to_trajectory
    return get_results(map_params_to_trajectory, input_dictionaries, mode; show_progress)
end

function save_results(path, results)
    results = deepcopy(results)
    # Make sure that the results are put on the CPU
    @save path results
end

function load_results(path)
    results = nothing
    @load path results
    return results
end

function run_and_save_mnist_problem(; kwargs...)
    results = mnist_trajectory_experiment(; kwargs...)
    try
        git_hash = get_git_hash()
        for r in results
            r[:git_hash] = git_hash
        end
    catch
        println("Was not able to get Git hash.")
    end

    save_results("results/large/mnist_data.bson", results)
end