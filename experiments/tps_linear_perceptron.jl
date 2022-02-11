using NNE.LinearTPSCalculation
using TPS
import BSON: @save
using Base.Iterators
using ProgressMeter
using Statistics
using Distributed

min_s = 0.01
max_s = 1000
num_s = 13
s_values = exp.((LinRange(log(min_s), log(max_s), num_s)))
t_values = [1, 2, 4, 8, 16]
σ = 1.0

function recreate_problem(problem::TPS.DiscreteTrajectory.DTProblem, new_state)
    observable = TPS.get_observable(problem)
    return TPS.DiscreteTrajectory.DTProblem(observable, new_state)
end
function recreate_problem(problem::TPS.SimulatedAnnealing.SAProblem, new_state)
    observable = TPS.get_observable(problem)
    return TPS.SimulatedAnnealing.SAProblem(observable, new_state)
end
function recreate_problem(problem, new_state)
    throw("Unimplemented problem type")
end


function solve_tps(s, τ)
    problem = construct_problem(τ, σ)
    # Need to actually think about this to get proper convergence for high sigma!
    epochs = 20000
    # Anneal for high s
    solution = nothing
    if s > 1.0
        alg = construct_algorithm(τ, 1.0, σ)
        solution = solve(problem, alg, epochs)
        anneal_steps = Int(round((log10(s)+1)*(2+log2(τ))))
        for k in 1:anneal_steps
            problem = recreate_problem(problem, TPS.get_current_state(solution))
            new_s = 10.0 ^ ((k / anneal_steps) * (log10(s) - log10(1.0)) + log10(1.0))
            alg = construct_algorithm(τ, new_s, σ)
            if new_s ≈ s
                solution = solve(problem, alg, epochs*10) # Run final s for a lot longer
            else
                solution = solve(problem, alg, epochs)
            end
        end
    else
        alg = construct_algorithm(τ, s, σ)
        solution = solve(problem, alg, epochs*5*τ)
    end

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
function get_results_distributed()
    iter = collect(product(s_values, t_values))
    results = @showprogress pmap(iter) do x
        solve_tps(x...)
    end
    return results
end

function get_avg_loss(solution; skip=0)
    return mean(solution.observations[(skip+1):end])
end

#-----RUNNING THE CODE-----#
# In order to run this code, you can get the results in threaded mode:
# results = get_results();
# or in a distributed way:
# results = get_results_distributed();

#-----SAVING THE CODE-----#
# losses = get_avg_loss.(results)
# @save "results/large/tps_linear_perceptron.bson" s_values t_values σ results losses