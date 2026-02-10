"""
Functions for creating TPS explanation figures showing trajectory perturbations.
"""

include("plotting_utilities.jl")

using TransitionPathSampling
using CairoMakie
using LaTeXStrings
using TransitionPathSampling.MetropolisHastings
using TransitionPathSampling.DiscreteTrajectory
using Random

"""
    generate_random_walk_trajectory(τ, σ; rng=Random.GLOBAL_RNG)

Generate a single parameter trajectory using random walk.
"""
function generate_random_walk_trajectory(τ, σ; rng=Random.GLOBAL_RNG)
    states = [[0.0]]
    for t = 2:τ
        push!(states, [states[t-1][begin] + σ * randn(rng)])
    end
    return states
end

"""
    plot_trajectory_on_axis!(ax, states; indices=(1:length(states)), kwargs...)

Plot a parameter trajectory on the given axis with scatterlines.
"""
function plot_trajectory_on_axis!(ax, states; indices=(1:length(states)), kwargs...)
    y_vals = [s[begin] for s in states]
    scatterlines!(ax, collect(indices), y_vals; kwargs...)
end

"""
    generate_tps_perturbations(states, σ)

Generate different types of TPS perturbations (forwards, backwards, bridge) for a given trajectory.
"""
function generate_tps_perturbations(states, σ)
    # Generate cache with dummy loss function
    loss_fn(x::AbstractArray) = 0.0
    loss_fn(x::AbstractArray{T}) where {T<:AbstractArray} = [loss_fn(y) for y in x]
    cache = TransitionPathSampling.generate_cache(
        MetropolisHastings.gaussian_trajectory_algorithm(0.0, σ),
        DiscreteTrajectory.DTProblem(TransitionPathSampling.SimpleObservable(loss_fn), states)
    )

    # Shooting forwards
    MetropolisHastings.shoot!(cache, states, 6, σ, true)
    forwards_state = deepcopy(cache)

    # Shooting backwards
    MetropolisHastings.shoot!(cache, states, 4, σ, false)
    backwards_state = deepcopy(cache)

    # Bridging
    MetropolisHastings.bridge!(cache, states, 3, 7, σ)
    bridge_state = deepcopy(cache)

    perturbed_states = Dict{Symbol,Any}(
        :forwards => forwards_state,
        :backwards => backwards_state,
        :bridge => bridge_state
    )
    return perturbed_states
end

"""
    plot_tps_perturbation_examples(seed=1141; border=0.05)

Create TPS explanation figure showing different types of trajectory perturbations.

This function generates a three-panel figure demonstrating:
- (a) Backward shooting perturbation
- (b) Forward shooting perturbation
- (c) Bridge perturbation

# Arguments
- `seed`: Random seed for reproducibility (default: 1141, alternatives: 1189)
- `border`: Border size as fraction of range for y-axis limits (default: 0.05)
"""
function plot_tps_perturbation_examples(seed=1141; border=0.05)
    σ = 1.0
    τ = 10
    Random.seed!(seed)
    states = generate_random_walk_trajectory(τ, σ)
    perturbed_states = generate_tps_perturbations(states, σ)

    # Calculate limits across all states
    minimum_value = minimum(first, states)
    maximum_value = maximum(first, states)

    for (key, cache) in perturbed_states
        new_state = deepcopy(states)
        MetropolisHastings.apply!(new_state, cache)
        maximum_value = max(maximum_value, maximum(first, new_state))
        minimum_value = min(minimum_value, minimum(first, new_state))
    end

    range_vals = maximum_value - minimum_value
    ylim_range = (minimum_value - border * range_vals, maximum_value + border * range_vals)

    # Create figure with 2 panels
    fig = create_publication_figure(num_panels=2, num_panels_y=1)

    original_color = :purple
    new_color = :coral

    titles = [L"(a)", L"(b)", L"(c)"]
    keys = [:backwards, :forwards, :bridge]

    for (i, key) in enumerate(keys)
        ax = Axis(fig[1, i],
            xlabel=L"t",
            ylabel=i == 1 ? L"\Theta" : "",
            title=titles[i],
            titlealign=:left,
            xticksvisible=false,
            yticksvisible=false,
            xticklabelsvisible=false,
            yticklabelsvisible=false,
            xgridvisible=false,
            ygridvisible=false,
            rightspinevisible=false,
            topspinevisible=false)

        cache = perturbed_states[key]

        # Plot original trajectory
        plot_trajectory_on_axis!(ax, states;
            color=original_color, linewidth=5.0, marker=:circle, markersize=24,
            strokecolor=:black, strokewidth=1, label=L"\omega")

        # Plot perturbed trajectory
        new_state = deepcopy(states)
        MetropolisHastings.apply!(new_state, cache)
        plot_trajectory_on_axis!(ax, new_state;
            color=new_color, linewidth=5.0, linestyle=(:dash, :dense),
            marker=:utriangle, markersize=20, strokecolor=:black, strokewidth=1, label=L"\omega'")

        ylims!(ax, ylim_range)
        axislegend(ax, position=:lt, framevisible=false)
    end

    return fig
end

"""
    save_tps_perturbation_examples_figure(seed=1141)

Create and save the TPS perturbation examples figure.
"""
function save_tps_perturbation_examples_figure(seed=1141)
    fig = plot_tps_perturbation_examples(seed)
    save(joinpath("figures", "perturbation_examples.pdf"), fig)
    return fig
end
