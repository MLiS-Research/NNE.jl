import BSON: @save
using Base.Iterators
using Distributed
using NNE
using NNE.Runner
using NNE.ToyClassificationProblem: construct_toy_problem, construct_algorithm as construct_toy_algorithm
using ProgressMeter
using Statistics
using TPS
using TPS.Annealing

include("run_helper.jl")

function solve_toy_tps(s, τ, σ; epochs = 10000)
    problem = construct_toy_problem(τ, σ)
    # Anneal for high s
    solution = nothing
    if s > 1.0
        backing_alg = construct_toy_algorithm(τ, 1.0, σ)
        anneal_steps = Int(round((log10(s) + 1) * (2 + log2(τ))))
        alg = create_exponential_decay_algorithm(backing_alg, 1.0, s, anneal_steps * epochs, :s; max_parameter_value = s)
        solution = solve(problem, alg, (anneal_steps + 10) * epochs)
    else
        alg = construct_toy_algorithm(τ, s, σ)
        solution = solve(problem, alg, epochs * 5 * τ)
    end

    return solution
end
function run_no_heuristics(s, τ, σ; epochs = 10000)
    problem = construct_toy_problem(τ, σ)
    alg = construct_toy_algorithm(τ, s, σ)
    solution = solve(problem, alg, epochs)
    return solution
end

function main(; execution_mode::TaskExecutionMode = SerialMode, show_progress = true)
    min_s = 0.01
    max_s = 5.0
    num_s = 8
    s_values = get_exponentially_spaced(min_s, max_s, num_s)
    t_values = [1, 2, 4, 8, 16]
    σ = 0.1
    epochs = 50000
    fn(x...) = solve_toy_tps(x...; epochs = epochs)
    iter = collect(product(s_values, t_values, [σ]))
    results = get_results(fn, iter, execution_mode; show_progress)
    @save "results/large/tps_toy_classification.bson" s_values t_values σ results
end

function get_file_path(id)
    if id > 0
        return "results/large/tps_toy_classification_run_$id.bson"
    else
        return "results/large/tps_toy_classification.bson"
    end
end

function load_tps_toy_data(; id = 0)
    s_values = nothing
    t_values = nothing
    σ = nothing
    results = nothing
    @load get_file_path(id) s_values t_values σ results
    results = results[:, :, 1] # Collapse the sigma dimension
    return s_values, t_values, σ, results
end

function rerun_from_save(; execution_mode::TaskExecutionMode = SerialMode, show_progress = true, epochs = 100000, initial_id = 0)
    s_values, t_values, σ, results = load_tps_toy_data(; id = initial_id)
    fn(x...) = run_no_heuristics(x...; epochs = epochs)
    iter = collect(product(s_values, t_values, [σ]))
    results = get_results(fn, iter, execution_mode; show_progress)
    id = 1
    while isfile(get_file_path(id))
        id += 1
    end
    @save get_file_path(id) s_values t_values σ results
end