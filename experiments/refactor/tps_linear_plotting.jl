"""
Functions for plotting TPS linear perceptron results.
"""

include("plotting_utilities.jl")

using Statistics
using BSON: @load, @save

"""
    load_raw_tps_linear_data()

Load raw TPS linear perceptron data from BSON file.
"""
function load_raw_tps_linear_data()
    s_values = nothing
    t_values = nothing
    σ = nothing
    losses = nothing
    @load "results/large/tps_linear_perceptron.bson" s_values t_values σ losses
    return s_values, t_values, σ, losses
end

"""
    load_processed_tps_linear_data()

Load processed TPS linear data from BSON file.
"""
function load_processed_tps_linear_data()
    s_values = nothing
    t_values = nothing
    σ = nothing
    losses = nothing
    @load "results/tps_linear_reduced_data.bson" s_values t_values σ losses
    return s_values, t_values, σ, losses
end

"""
    process_and_save_tps_linear_data()

Process TPS linear data by averaging the last 75% of losses and save reduced data.
"""
function process_and_save_tps_linear_data()
    s_values, t_values, σ, losses = load_raw_tps_linear_data()
    times_arr = hcat(repeat(t_values', length(s_values)))
    get_last_n_losses(losses, n) = mean(losses[(end-n):end])
    losses = (x -> get_last_n_losses(x, Int(round(0.75 * length(x))))).(losses) ./ times_arr
    losses = reshape(losses, length(s_values), length(t_values))

    @save "results/tps_linear_reduced_data.bson" s_values t_values σ losses
    nothing
end

"""
    plot_tps_linear_perceptron(; kwargs...)

Construct a plot of TPS data loss vs s parameter.
"""
function plot_tps_linear_perceptron(; kwargs...)
    s_values, t_values, _, losses = load_processed_tps_linear_data()

    markers = [:ltriangle, :diamond, :rect, :dtriangle, :circle]
    fig = plot_loss_vs_s_parameter(s_values, t_values, losses;
        markershape=markers,
        kwargs...
    )

    return fig
end

"""
    save_tps_linear_perceptron_figure()

Create and save the TPS linear perceptron figure.
"""
function save_tps_linear_perceptron_figure()
    fig = plot_tps_linear_perceptron()
    save(joinpath("figures", "tps_linear_perceptron.pdf"), fig)
    return fig
end
