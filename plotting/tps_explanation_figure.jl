using TPS
using Plots
using LaTeXStrings
using TPS.MetropolisHastings
using TPS.DiscreteTrajectory
using Random
include("plotting_style.jl")

function generate_single_parameter_problem(τ, σ; rng=Random.GLOBAL_RNG)
    states = [[0.0]]
    for t=2:τ
        push!(states, [states[t-1][begin]+σ*randn(rng)])
    end
    return states
end

function plot_parameter_trajectory(states; new_plot=true, kwargs...)
    plot_fn = new_plot ? plot : plot!
    plt = plot_fn(0:length(states)-1, [s[begin] for s in states], legend=false; kwargs...)
    xlabel!(L"t")
    ylabel!(L"\theta")
    return plt
end

function generate_perturbations(states, σ; rng=Random.GLOBAL_RNG)
    forwards_perturbation = TPS.MetropolisHastings.shoot_perturbation(states, 6, σ, true; rng=rng)
    backwards_perturbation = TPS.MetropolisHastings.shoot_perturbation(states, 4, σ, false; rng=rng)
    bridge_perturbation = TPS.MetropolisHastings.bridge_perturbation(states, 3, 7, σ; rng=rng)

    perturbations = Dict{Symbol, Any}(:forwards=>forwards_perturbation, :backwards=>backwards_perturbation, :bridge=>bridge_perturbation)
    return perturbations
end

function main_plot(seed=1234)
    rng = Random.MersenneTwister(seed)
    σ=1.0
    τ=10
    states = generate_single_parameter_problem(τ, σ; rng=rng)
    perturbations = generate_perturbations(states, σ; rng=rng)
    function get_trajectory(p)
        changes, indices = p
        new_trajectory = deepcopy(states)
        new_trajectory[indices] .+= changes
        return new_trajectory
    end

    plots = Dict{Symbol, Any}()
    for (key, value) in perturbations
        plt = plot_parameter_trajectory(states, label=L"\omega", markershape=:utriangle)
        traj = get_trajectory(value)
        plot_parameter_trajectory(traj; new_plot=false, label=L"\omega'", markershape=:circle)
        
        plot!(;yticks=false, xticks=false, legend=:topleft)
        # if key != :bridge
        #     plot!(;legend=false)
        #     ylabel!("") # Turn off the y label
        # else
        #     plot!(;legend=:topleft)
        # end

        plots[key] = plt
    end

    layout = @layout [a b c]

    plot_defaults = get_plot_defaults_full_width()

    plt = plot(plots[:backwards], plots[:forwards], plots[:bridge]; layout=layout, title=["(a)" "(b)" "(c)"], link=:y, titleloc=:left, plot_defaults...)

    savefig(plt, "figures/bridge_explanation_figure.pdf")
    return plt
end
