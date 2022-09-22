"""
This module contains code examples for generating a toy 2D loss function and showing how TPS works.
"""
module ToyProblem

using Random
using TransitionPathSampling
using TransitionPathSampling.DiscreteTrajectory
using TransitionPathSampling.MetropolisHastings

include("../helpers/setup.jl")

export toy_loss_fn, create_toy_problem, create_tps_algorithm, create_tps_solution, iterate_solution!, convert_trajectory_to_matrix

"""
    toy_loss_fn(x, y)

Takes in 2D coordinates and returns a loss at that point.

The domain of x, y are bounded to [-5.0, 5.0].

This is Hummelblau's function, with four global optima at [3.0, 2.0], [-2.805118, 3.131312],
[-3.779310, -3.283186] and [3.584428, -1.848126].
"""
function toy_loss_fn(x, y)
    return (x * x + y - 11)^2 + (x + y * y - 7)^2
end

"""
    generate_parameters()

Generates a vector of 2 parameters in the domain of [-1.0, 1.0]
"""
generate_parameters() = [rand() * 2 - 1, rand() * 2 - 1]

function create_toy_problem(τ, σ; rng=Random.GLOBAL_RNG)
    @assert τ > 1 "Make sure that the trajectory length is greater than 1."
    initial_params = generate_parameters()
    initial_state = generate_trajectory(initial_params, τ, σ; rng)

    loss_fn_single(state) = toy_loss_fn(state...)
    loss_fn(state) = sum(loss_fn_single.(state))
    obs = TransitionPathSampling.SimpleObservable(loss_fn)
    problem = DTProblem(obs, initial_state)

    return problem
end

function create_tps_algorithm(τ, s, σ; rng=Random.GLOBAL_RNG)
    if τ < 4
        return get_shooting_mh_alg(s, σ; rng)
    else
        get_shooting_and_bridging_mh_alg(s, σ; rng, max_width=4)
    end
end

create_tps_solution(problem, algorithm) = init_solution(algorithm, problem)

function iterate_solution!(solution, algorithm, epochs)
    for i = 1:epochs
        step!(solution, algorithm, i)
    end
    nothing
end

"""
    convert_trajectory_to_matrix(states)

Converts a trajectory of states into a matrix, with the first dimension being τ
and the second being the number of parameters.
"""
convert_trajectory_to_matrix(states) = hcat(states...)'

end