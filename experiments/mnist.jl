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
using TPS.Convergence


function get_experiment_parameter_dictionaries()
    s_min = 0.01
    s_max = 5.0
    num_s_values = 11
    s_values = 10.0 .^ (LinRange(log10(s_min), log10(s_max), num_s_values))
    trajectory_lengths = [1, 2, 4, 8, 16]
    options = Dict{Symbol, Any}()
    options[:fraction_to_include] = 0.25
    options[:warmup_steps] = 75000
    options[:polling_frequency] = 100
    options[:max_buffer_size] = 400000
    options[:max_epochs] = 500000
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

function mnist_trajectory_experiment(; execution_mode=:serial)
    input_dictionaries = get_experiment_parameter_dictionaries()

    # Threaded
    if execution_mode==:threaded
        results = convert(Matrix{Any}, similar(input_dictionaries))
        progress = Progress(length(results))
        Threads.@threads for i in 1:length(results)
            results[i] = map_params_to_trajectory(input_dictionaries[i])
            GC.gc()
            CUDA.reclaim()
            next!(progress)
        end
        return results
    end

    if execution_mode==:distributed
        # Distributed
        results = @showprogress pmap(input_dictionaries) do dict
            map_params_to_trajectory(dict)
        end
        return results
    end

    if execution_mode==:serial
        results = convert(Matrix{Any}, similar(input_dictionaries))
        progress = Progress(length(results))
        for i in 1:length(results)
            results[i] = map_params_to_trajectory(input_dictionaries[i])
            next!(progress)
        end
        return results
    end

    error("Execution mode is not implemented: $execution")
end

function save_results(path, results)
    results = deepcopy(results)
    # Make sure that the results are put on the CPU
    @save path results
end

function run_mnist_problem(; execution_mode=:serial)
    results = mnist_trajectory_experiment(;execution_mode)

    save_results("results/large/mnist_data_1.bson", results)
end