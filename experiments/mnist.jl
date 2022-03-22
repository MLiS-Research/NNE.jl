using Distributed
@everywhere include("distributed/mnist_runner.jl")
import Base.Iterators: product
using BSON: @save, @load, parse
using ProgressMeter
using Statistics
using Flux
using CUDA
using TPS
using NNE
using NNE.Runner
using TPS.Convergence


function get_experiment_parameter_dictionaries(;device=gpu)
    s_min = 0.01
    s_max = 5.0
    num_s_values = 11
    s_values = 10.0 .^ (LinRange(log10(s_min), log10(s_max), num_s_values))
    trajectory_lengths = [1, 2, 4, 8, 16]
    options = Dict{Symbol, Any}()
    options[:fraction_to_include] = 0.25
    options[:warmup_steps] = 5000
    options[:polling_frequency] = 100
    options[:max_buffer_size] = 400000
    options[:max_epochs] = 20000
    options[:relative_gradient_size] = 1e-7
    options[:relative_error_size] = 1e-3
    options[:device] = device
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

function mnist_trajectory_experiment(; mode::TaskExecutionMode=SerialMode, device=gpu, show_progress=false)
    input_dictionaries = get_experiment_parameter_dictionaries(;device)
    map_params_to_trajectory
    return get_results(map_params_to_trajectory, input_dictionaries; mode, show_progress)
end

function save_results(path, results)
    results = deepcopy(results)
    # Make sure that the results are put on the CPU
    @save path results
end

function run_and_save_mnist_problem(; kwargs...)
    results = mnist_trajectory_experiment(; kwargs...)

    save_results("results/large/mnist_data_test.bson", results)
end