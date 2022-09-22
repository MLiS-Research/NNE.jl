using Base.Iterators
using NNE.LinearTPSCalculation
using Statistics
using TransitionPathSampling
using TransitionPathSampling.Annealing
using Dates

function linear_tps(config, trial_id)
    return _solve_linear_tps(; config...)
end

function _solve_linear_tps(; s, τ, σ, annealing_epochs=10000, epochs=10000, max_width=3, initial_s=1.0, kwargs...)

    results = Dict{Symbol,Any}(:s => s, :τ => τ, :σ => σ, :annealing_epochs => annealing_epochs, :epochs => epochs)
    results[:start_time] = now()

    problem = construct_problem(τ, σ)

    results[:initial_state] = TransitionPathSampling.get_initial_state(problem)
    if s > 1.0 # anneal for high s
        backing_alg = construct_algorithm(τ, initial_s, σ; max_width=max_width)
        alg = LinearDecayAnnealedAlgorithm(backing_alg, initial_s, s, annealing_epochs, :s)
        solution = solve(problem, alg, epochs + annealing_epochs)
        results[:losses] = deepcopy(solution.observations)
        results[:final_state] = get_current_state(solution)
    else
        alg = construct_algorithm(τ, s, σ)
        solution = solve(problem, alg, epochs + annealing_epochs)
        results[:losses] = deepcopy(solution.observations)
        results[:final_state] = get_current_state(solution)
    end

    results[:end_time] = now()
    results[:duration] = results[:end_time] - results[:start_time]
    results[:git_hash] = get_git_hash()
    results[:worker_id] = Distributed.myid()

    return results
end

function get_git_hash()
    return strip(read(`git rev-parse HEAD`, String))
end