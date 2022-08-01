import BSON: @save
using Base.Iterators
using Distributed
using NNE.LinearTPSCalculation
using NNE.Runner
using ProgressMeter
using Statistics
using TPS
using TPS.Annealing

include("run_helper.jl")


function solve_linear_tps(s, τ; epoch_block_size=20000)
    problem = construct_problem(τ, σ)
    # Anneal for high s
    solution = nothing
    if s > 1.0
        backing_alg = construct_algorithm(τ, 1.0, σ)
        anneal_steps = Int(round((log10(s)+1)*(2+log2(τ))))
        alg = create_exponential_decay_algorithm(backing_alg, 1.0, s, anneal_steps*epoch_block_size, :s; max_parameter_value=s)
        solution = solve(problem, alg, (anneal_steps+10)*epoch_block_size)
    else
        alg = construct_algorithm(τ, s, σ)
        solution = solve(problem, alg, epoch_block_size*5*τ)
    end

    return solution
end

function get_avg_loss(solution; skip=0)
    return mean(solution.observations[(skip+1):end])
end

function main(;execution_mode::TaskExecutionMode=SerialMode, show_progress=true)
    min_s = 0.01
    max_s = 1000
    num_s = 13
    s_values = get_exponentially_spaced(min_s, max_s, num_s)
    t_values = [1, 2, 4, 8, 16]
    σ = 1.0
    epoch_block_size=20000
    fn(x...) = solve_linear_tps(x...; epoch_block_size)
    iter = collect(product(s_values, t_values, [σ]))
    results = get_results(fn, iter, execution_mode; show_progress)
    losses = get_avg_loss.(results)
    @save "experiments/results/large/tps_linear_perceptron.bson" s_values t_values σ results losses
end