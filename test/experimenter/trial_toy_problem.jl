using NNE
using NNE.ToyClassificationProblem
using NNE.CallbackGenerator: create_callbacks
using TPS
using Flux


function run_problem(config, trial_id)
    τ = config[:τ]
    s = config[:s]
    σ = config[:σ]
    epochs = config[:epochs]
    info = Dict{Symbol, Any}()
    problem = construct_toy_problem(τ, σ)
    algorithm = construct_algorithm(τ, s, σ)
    cb = create_callbacks(trial_id, identity, info; snapshot_every_n=5, snapshot_label="T = $τ, s = $s, sigma = $σ", save_final_snapshot=true, use_previous_snapshot=true)

    sol = solve(problem, algorithm, 1:epochs; cb=cb)

    info[:observations] = collect(sol)

    return info
end