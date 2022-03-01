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

function solve_toy_tps(s, τ, σ; epochs=10000)
    problem = construct_toy_problem(τ, σ)
    # Anneal for high s
    solution = nothing
    if s > 1.0
        backing_alg = construct_toy_algorithm(τ, 1.0, σ)
        anneal_steps = Int(round((log10(s)+1)*(2+log2(τ))))
        alg = create_exponential_decay_algorithm(backing_alg, 1.0, s, anneal_steps*epochs, :s; max_parameter_value=s)
        solution = solve(problem, alg, (anneal_steps+10)*epochs)
    else
        alg = construct_toy_algorithm(τ, s, σ)
        solution = solve(problem, alg, epochs*5*τ)
    end

    return solution
end

function main(;execution_mode::TaskExecutionMode=SerialMode, show_progress=true)
    min_s = 0.01
    max_s = 5.0
    num_s = 8
    s_values = get_exponentially_spaced(min_s, max_s, num_s)
    t_values = [1, 2, 4, 8, 16]
    σ = 0.1
    epochs = 50000
    fn(x...) = solve_toy_tps(x...; epochs=epochs)
    iter = collect(product(s_values, t_values, [σ]))
    results = get_results(fn, iter, execution_mode; show_progress)
    @save "results/large/tps_toy_classification.bson" s_values t_values σ results
end