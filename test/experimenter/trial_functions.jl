using TransitionPathSampling
using TransitionPathSampling.MetropolisHastings
using TransitionPathSampling.DiscreteTrajectory
using TransitionPathSampling.SimulatedAnnealing
using NNE
using NNE.Experimenter
using NNE.CallbackGenerator

function run_experiment(config, trial_id)
    info = Dict{Symbol,Any}()

    info[:value] = config[:n] * config[:m]
    info[:config] = config
    info[:trial_id] = trial_id

    return info
end

function run_restore_experiment(config, trial_id)
    info = Dict{Symbol,Any}()
    problem = example_tps_problem(config[:τ], config[:d])
    if haskey(config, :restore_from_trial_id)
        restore_state!(problem, config[:restore_from_trial_id])
    end
    info[:initial_state] = TransitionPathSampling.get_initial_state(problem)
    alg = example_tps_algorithm(config[:s], config[:σ], config[:τ])
    sol = solve(problem, alg, 1:config[:epochs])
    info[:final_state] = get_current_state(sol)

    return info
end


square(x) = x * x
function example_tps_loss_fn(state::AbstractArray{T}) where {T<:AbstractArray}
    return [example_tps_loss_fn(s) for s in state]
end
function example_tps_loss_fn(state::AbstractArray)
    return sum(square, state) / length(state)
end

function example_tps_problem(τ, d)
    states = [rand(d) for _ in 1:τ]
    obs = TransitionPathSampling.SimpleObservable(example_tps_loss_fn)
    return τ == 1 ? SAProblem(obs, first(states)) : DTProblem(obs, states)
end

function example_tps_algorithm(s, σ, τ)
    if τ == 1
        return gaussian_sa_algorithm(s, σ)
    else
        return gaussian_trajectory_algorithm(s, σ)
    end
end