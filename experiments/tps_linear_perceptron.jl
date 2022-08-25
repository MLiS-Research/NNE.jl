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


function solve_linear_tps(s, τ, σ; epoch_block_size=20000, min_epochs=0)
    problem = construct_problem(τ, σ)
    # Anneal for high s
    solution = nothing
    if s > 1.0
        backing_alg = construct_algorithm(τ, 1.0, σ; max_width=4)
        anneal_steps = Int(round((log10(s)+1)*(2+log2(τ))))
        alg = LinearDecayAnnealedAlgorithm(backing_alg, 1.0, s, anneal_steps*epoch_block_size, :s)
        solution = solve(problem, alg, min_epochs+(anneal_steps+5)*epoch_block_size)
    else
        alg = construct_algorithm(τ, s, σ)
        solution = solve(problem, alg, min_epochs+epoch_block_size*10*τ)
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
    epoch_block_size=20_000
    fn(x...) = solve_linear_tps(x...; epoch_block_size, min_epochs=Int(1e6))
    iter = collect(product(s_values, t_values, [σ]))
    results = get_results(fn, iter, execution_mode; show_progress)
    losses = (x->Float32.(x.observations)).(results)
    @save "results/large/tps_linear_perceptron.bson" s_values t_values σ losses
end