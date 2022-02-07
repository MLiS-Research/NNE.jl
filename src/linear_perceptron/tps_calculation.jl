module LinearTPSCalculation
using TPS
using TPS.SimulatedAnnealing
using TPS.MetropolisHastings
using TPS.Convergence
using TPS.DiscreteTrajectory

include("data_generation.jl")

get_flat_weights(rng) = reshape(generate_weights(rng, 1, 1), :) 

function create_sa_problem(loss_fn; rng=Random.GLOBAL_RNG)
    initial_state = get_flat_weights(rng)
    obs = TPS.SimpleObservable(loss_fn)
    return SAProblem(obs, initial_state)
end
function create_trajectory_problem(loss_fn, τ, σ; rng=Random.GLOBAL_RNG)
    initial_state = get_flat_weights(rng)
    states = [initial_state]
    for t in 2:τ
        next_state = states[t-1] .+ σ .* randn(rng, size(initial_state)...)
        push!(states, next_state)
    end

    trajectory_loss_fn(state) = sum(loss_fn(s) for s in state)
    obs = TPS.SimpleObservable(trajectory_loss_fn)
    return DTProblem(obs, states)
end

function construct_problem(τ, σ=1.0, seed=1234)
    x,y,_ = generate_seeded_data(seed)
    rng = Random.MersenneTwister(seed*4)
    # close the loss function with the inputs and outputs
    loss_fn(w) = loss(reshape(w, 1, :), x, y)

    problem = τ==1 ? create_sa_problem(loss_fn;rng) : create_trajectory_problem(loss_fn, τ, σ; rng)
    return problem
end

function construct_algorithm(τ, s, σ)
    if τ==1
        return get_guassian_mh_alg(s, σ)
    else
        return MetropolisHastings.get_shooting_mh_alg(s, σ)
    end
end

export construct_algorithm, construct_problem
end