module ProblemGenerators
using TransitionPathSampling
using TransitionPathSampling.SimulatedAnnealing
using TransitionPathSampling.DiscreteTrajectory
using Random

function create_sa_problem(loss_fn, initial_state)
    obs = TransitionPathSampling.SimpleObservable(loss_fn)
    return SAProblem(obs, initial_state)
end
function create_trajectory_problem(loss_fn, initial_state, τ, σ; rng=Random.GLOBAL_RNG)
    states = [initial_state]
    for t in 2:τ
        next_state = states[t-1] .+ σ .* randn(rng, size(initial_state)...)
        push!(states, next_state)
    end

    function trajectory_loss_fn(state)
        return loss_fn(state)
    end
    function trajectory_loss_fn(states::AbstractArray{T}) where {T<:AbstractArray}
        return [loss_fn(state) for state in states]
    end

    obs = TransitionPathSampling.SimpleObservable(trajectory_loss_fn)
    return DTProblem(obs, states)
end
function create_problem(loss_fn, initial_state, τ, σ; rng=Random.GLOBAL_RNG)
    if τ == 1
        return create_sa_problem(loss_fn, initial_state)
    else
        return create_trajectory_problem(loss_fn, initial_state, τ, σ; rng=rng)
    end
end

export create_problem, create_sa_problem, create_trajectory_problem

end