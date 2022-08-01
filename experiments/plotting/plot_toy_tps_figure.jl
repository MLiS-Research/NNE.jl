using NNE
using NNE.ToyProblem
using Random
using TPS
include("plotting_style.jl")

function plot_loss_function(;kwargs...)    
    n = 1024
    x = LinRange(-4.8, 4.8, n)
    plt = contour(x, x, (a, b)->log10(toy_loss_fn(a,b)); colorbar=nothing, kwargs...)
    xlabel!("x")
    ylabel!("y")
    return plt
end

function plot_experiments(states, losses)
    num_states = length(states)
    @assert length(losses)==num_states "Must have the same number of losses as states"

    plts = [plot!(plot_loss_function(), (x->first(x)).(s), (x->last(x)).(s), legend=false; lw=3, markershape=:circle, color=:blue) for s in states]
    for (p, lbl) in zip(plts, collect('a':'z')[1:length(plts)])
        plot!(p; ticks=false, showaxis=false, xlims=(-5.0, 5.0), ylims=(-5.0, 5.0))
        xlabel!(p, "")
        ylabel!(p, "")
        title!(p, "($lbl)")
    end
    plt = plot(plts...; dpi=300)

    return plt
end


"""
    run_experiment(τ, s, σ; num_steps, step_size, seed)
"""
function run_experiment(τ, s, σ; num_steps=10, step_size=10, seed=1234)
    rng = Random.MersenneTwister(seed)
    problem = create_toy_problem(τ, σ; rng)
    algorithm = create_tps_algorithm(τ, s, σ; rng)
    solution = create_tps_solution(problem, algorithm)

    states = [deepcopy(TPS.get_current_state(solution))]
    epochs = [1]
    for _ = 1:num_steps
        iterate_solution!(solution, algorithm, step_size)
        push!(epochs, last(epochs)+step_size)
        push!(states, deepcopy(TPS.get_current_state(solution)))
    end

    return states, collect(solution)[epochs]
end