using TPS
using Plots
using LaTeXStrings
using TPS.MetropolisHastings
using TPS.DiscreteTrajectory
using Random
include("plotting_style.jl")

function generate_single_parameter_problem(τ, σ; rng=Random.GLOBAL_RNG)
    states = [[0.0]]
    for t = 2:τ
        push!(states, [states[t-1][begin] + σ * randn(rng)])
    end
    return states
end

function plot_parameter_trajectory(states; indices=(1:length(states)), new_plot=true, kwargs...)
    plot_fn = new_plot ? plot : plot!
    plt = plot_fn(indices, [s[begin] for s in states], legend=false; alpha=1.0, kwargs...)
    xlabel!(L"t")
    ylabel!(L"\Theta")
    return plt
end

function generate_perturbations(states, σ)
    # Generate initial state
    loss_fn(x::AbstractArray) = 0.0
    loss_fn(x::AbstractArray{T}) where {T<:AbstractArray} = [loss_fn(y) for y in x]
    cache = TPS.generate_cache(TPS.MetropolisHastings.gaussian_trajectory_algorithm(0.0, σ), TPS.DiscreteTrajectory.DTProblem(TPS.SimpleObservable(loss_fn), states))

    # Shooting forwards
    TPS.MetropolisHastings.shoot!(cache, states, 6, σ, true)
    forwards_state = deepcopy(cache)

    # Shooting backwards
    TPS.MetropolisHastings.shoot!(cache, states, 4, σ, false)
    backwards_state = deepcopy(cache)

    # Bridging
    TPS.MetropolisHastings.bridge!(cache, states, 3, 7, σ)
    bridge_end = deepcopy(cache)

    perturbed_states = Dict{Symbol,Any}(:forwards => forwards_state, :backwards => backwards_state, :bridge => bridge_end)
    return perturbed_states
end

function main_plot(seed=1141; border=0.025)
    # 1189, 1141 are good seed choices
    σ = 1.0
    τ = 10
    Random.seed!(seed)
    states = generate_single_parameter_problem(τ, σ)
    perturbed_states = generate_perturbations(states, σ)

    plots = Dict{Symbol,Any}()
    minimum_value = minimum(first, states)
    maximum_value = maximum(first, states)
    defaults = get_plot_defaults()

    original_color = palette(:matter)[192]
    new_color = palette(:matter)[64]
    for (key, cache) in perturbed_states
        plt = plot_parameter_trajectory(states; label=L"\omega", markershape=:circle, c=original_color)
        new_state = deepcopy(states)
        TPS.MetropolisHastings.apply!(new_state, cache)
        maximum_value = max(maximum_value, maximum(first, new_state))
        minimum_value = min(minimum_value, minimum(first, new_state))
        plot_parameter_trajectory(new_state; new_plot=false, label=L"\omega'", markershape=:utriangle, linestyle=:dash, c=new_color, defaults...)

        plot!(; yticks=false, xticks=false, legend=:topleft)

        plots[key] = plt
    end

    range_vals = maximum_value - minimum_value
    ylims!(minimum_value - border * range_vals, maximum_value + border * range_vals)

    layout = @layout [a b c]

    plot_defaults = get_plot_defaults(; columns=2, height_ratio=1 / 4)
    plt = plot(plots[:backwards], plots[:forwards], plots[:bridge]; layout=layout, title=[L"(a)" L"(b)" L"(c)"], link=:y, titleloc=:left, plot_defaults...)

    savefig(plt, joinpath("figures", "perturbation_examples.pdf"))
    return plt
end
