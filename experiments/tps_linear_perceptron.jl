using NNE.LinearTPSCalculation
using TPS
import BSON: @save
using Base.Iterators
using ProgressMeter
using Statistics

min_s = 0.01
max_s = 10000
num_s = 11
s_values = exp.((LinRange(log(min_s), log(max_s), num_s)))
t_values = [1, 2, 4, 8, 16]
σ = 1.0


function solve_tps(s, τ)
    problem = construct_problem(τ, σ)
    alg = construct_algorithm(τ, s, σ)
    # Need to actually think about this to get proper convergence for high sigma!
    solution = solve(problem, alg, 1000000)

    return solution
end

function get_results()
    iter = collect(enumerate(product(s_values, t_values)))
    results = Array{Any}(undef, size(iter)...)
    progress = Progress(length(iter))
    Threads.@threads for (i, x) in iter
        results[i] = solve_tps(x...)
        next!(progress)
    end
    return results
end

function get_avg_loss(solution; skip=0)
    return mean(solution.observations[(skip+1):end])
end

results = get_results();
losses = get_avg_loss.(results; skip=100000)
@save "results/tps_linear_perceptron_all.bson" s_values t_values σ results losses