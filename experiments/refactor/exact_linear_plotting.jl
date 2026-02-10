"""
Functions for plotting exact linear perceptron results.
"""

include("plotting_utilities.jl")

using BSON: @load

"""
    load_exact_linear_perceptron_data()

Load exact linear perceptron data from BSON file.
"""
function load_exact_linear_perceptron_data()
    s_values = nothing
    t_values = nothing
    sigma = nothing
    problem = nothing
    losses = nothing
    @load "results/exact_linear_perceptron.bson" s_values t_values sigma problem losses
    return s_values, t_values, sigma, problem, losses
end

"""
    plot_exact_linear_perceptron(; min_s=nothing, max_s=nothing, results=load_exact_linear_perceptron_data(), kwargs...)

Construct a plot of loss vs s parameter for exact linear data.
"""
function plot_exact_linear_perceptron(; min_s=nothing, max_s=nothing, results=load_exact_linear_perceptron_data(), kwargs...)
    s_values, t_values, _, _, losses = results

    # Filter data if bounds specified
    if !isnothing(max_s) || !isnothing(min_s)
        selection_indices = s_values .>= (isnothing(min_s) ? typemin(eltype(s_values)) : min_s)
        selection_indices .&= s_values .<= (isnothing(max_s) ? typemax(eltype(s_values)) : max_s)
        s_values = s_values[selection_indices]
        losses = losses[selection_indices, :]
    end

    fig = plot_loss_vs_s_parameter(s_values, t_values, losses;
        ticks_kwargs=Dict(:round_digits => 0, :power_step => 2),
        linestyle=[:solid, :dash, :dot, :dashdot, :dashdotdot],
        kwargs...
    )

    return fig
end

"""
    save_exact_linear_perceptron_figure()

Create and save the exact linear perceptron figure.
"""
function save_exact_linear_perceptron_figure()
    fig = plot_exact_linear_perceptron()
    save(joinpath("figures", "exact_linear_perceptron.pdf"), fig)
    return fig
end
