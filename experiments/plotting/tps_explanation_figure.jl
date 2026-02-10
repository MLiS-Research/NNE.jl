using TransitionPathSampling
# using Plots
using CairoMakie
using LaTeXStrings
using TransitionPathSampling.MetropolisHastings
using TransitionPathSampling.DiscreteTrajectory
using Random
using DataFrames
using Measures: cm, mm, inch
include("plotting_style.jl")

Makie = CairoMakie

default_dpi() = 144
default_fontsize() = 10
function create_pub_fig(; dpi=default_dpi(), fontsize=default_fontsize(), num_panels=1, num_panels_y=1, kwargs...)
    resolution = Int.(round.((8.6cm * num_panels, 8.6cm * 21 / 28 * num_panels_y) ./ (1inch) .* dpi))
    pt_in_mm = 0.352777777777778mm
    font_height = Int(round(fontsize * pt_in_mm / 1inch * dpi))
    f = Figure(; fontsize=font_height, fonts=(; regular="Computer Modern"), resolution, dpi, kwargs...)
    return f
end
function get_marker_shape_dict(tau_values::AbstractArray{Int})
    possible_markers = [:circle, :diamond, :rect, :utriangle, :start4, :xcross]

    mapping = Dict{Int,Symbol}(
        t => m for (t, m) in Iterators.zip(tau_values, possible_markers)
    )
    return mapping
end
function get_marker_shape_dict(df::DataFrame)
    taus = sort(unique(df[!, :tau]))
    return get_marker_shape_dict(taus)
end

function generate_single_parameter_problem(τ, σ; rng=Random.GLOBAL_RNG)
    states = [[0.0]]
    for t = 2:τ
        push!(states, [states[t-1][begin] + σ * randn(rng)])
    end
    return states
end

function plot_parameter_trajectory!(ax, states; indices=(1:length(states)), kwargs...)
    y_vals = [s[begin] for s in states]
    # Makie.lines!(ax, collect(indices), y_vals; kwargs...)
    Makie.scatterlines!(ax, collect(indices), y_vals; kwargs...)
end

function generate_perturbations(states, σ)
    # Generate initial state
    loss_fn(x::AbstractArray) = 0.0
    loss_fn(x::AbstractArray{T}) where {T<:AbstractArray} = [loss_fn(y) for y in x]
    cache = TransitionPathSampling.generate_cache(MetropolisHastings.gaussian_trajectory_algorithm(0.0, σ), DiscreteTrajectory.DTProblem(TransitionPathSampling.SimpleObservable(loss_fn), states))

    # Shooting forwards
    MetropolisHastings.shoot!(cache, states, 6, σ, true)
    forwards_state = deepcopy(cache)

    # Shooting backwards
    MetropolisHastings.shoot!(cache, states, 4, σ, false)
    backwards_state = deepcopy(cache)

    # Bridging
    MetropolisHastings.bridge!(cache, states, 3, 7, σ)
    bridge_end = deepcopy(cache)

    perturbed_states = Dict{Symbol,Any}(:forwards => forwards_state, :backwards => backwards_state, :bridge => bridge_end)
    return perturbed_states
end

function main_plot(seed=1141; border=0.05)
    # 1189, 1141 are good seed choices
    σ = 1.0
    τ = 10
    Random.seed!(seed)
    states = generate_single_parameter_problem(τ, σ)
    perturbed_states = generate_perturbations(states, σ)

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

    # Create figure with 3 panels
    fig = create_pub_fig(num_panels=3, num_panels_y=1)

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
        plot_parameter_trajectory!(ax, states; color=original_color, linewidth=5.0, marker=:circle, markersize=24, strokecolor=:black, strokewidth=1, label=L"\omega")

        # Plot perturbed trajectory
        new_state = deepcopy(states)
        MetropolisHastings.apply!(new_state, cache)
        plot_parameter_trajectory!(ax, new_state; color=new_color, linewidth=5.0, linestyle=(:dash, :dense), marker=:utriangle, markersize=20, strokecolor=:black, strokewidth=1, label=L"\omega'")

        Makie.ylims!(ax, ylim_range)

        axislegend(ax, position=:lt, framevisible=false)
    end

    save(joinpath("figures", "perturbation_examples.pdf"), fig)
    return fig
end
