using Distributed
@everywhere include("distributed/mnist_runner.jl")
import Base.Iterators: product
using BSON: @save, @load, parse
using ProgressMeter
using Statistics
using Flux
using CUDA
using TPS
using TPS.Convergence


function get_experiment_parameter_dictionaries()
    s_min = 0.01
    s_max = 5.0
    num_s_values = 11
    s_values = 10.0 .^ (LinRange(log10(s_min), log10(s_max), num_s_values))
    trajectory_lengths = [1, 2, 4, 8, 16]
    options = Dict{Symbol, Any}()
    options[:fraction_to_include] = 0.25
    options[:warmup_steps] = 100
    options[:polling_frequency] = 10
    options[:max_buffer_size] = 10000
    options[:max_epochs] = 1000
    options[:relative_gradient_size] = 1e-7
    options[:relative_error_size] = 1e-3
    options[:device] = gpu
    options[:n_samples] = 512
    options[:outputs] = 2
    function construct_dict(s, τ)
        new_options = deepcopy(options)
        new_options[:s] = s
        new_options[:τ] = τ
        return new_options
    end
    input_dictionaries = [construct_dict(s, τ) for (s, τ) in product(s_values, trajectory_lengths)]
    return input_dictionaries
end

function mnist_trajectory_experiment(; use_local_execution=false)
    input_dictionaries = get_experiment_parameter_dictionaries()

    
    results = @showprogress pmap(input_dictionaries; distributed=use_local_execution) do dict
        map_params_to_trajectory(dict)
    end

    return results
end

function save_results(path, results)
    results = dict_to_cpu.(deepcopy(results))
    # Make sure that the results are put on the CPU
    @save path results
end

function run_mnist_problem(; use_local_execution=false)
    results = mnist_trajectory_experiment(;use_local_execution)

    save_results("results/large/mnist_data.bson", results)
end