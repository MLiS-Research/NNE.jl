include("plotting_style.jl")
include("plotting_utilities.jl")
using NNE
using TransitionPathSampling
using Random
using CairoMakie
using Statistics
using BSON: @load, @save
include("exact_linear_plotting.jl")

function load_tps_linear_data()
    s_values = nothing
    t_values = nothing
    σ = nothing
    losses = nothing
    @load "results/large/tps_linear_perceptron.bson" s_values t_values σ losses
    return s_values, t_values, σ, losses
end

function process_tps_linear_data_and_save()
    s_values, t_values, σ, losses = load_tps_linear_data()
    times_arr = hcat(repeat(t_values', length(s_values)))
    get_last_n_losses(losses, n) = mean(losses[(end-n):end])
    losses = (x -> get_last_n_losses(x, Int(round(0.75 * length(x))))).(losses) ./ times_arr
    losses = reshape(losses, length(s_values), length(t_values))

    @save "results/tps_linear_reduced_data.bson" s_values t_values σ losses
    nothing
end

function load_processed_tps_data()
    s_values = nothing
    t_values = nothing
    σ = nothing
    losses = nothing
    @load "results/tps_linear_reduced_data.bson" s_values t_values σ losses
    return s_values, t_values, σ, losses
end

function get_avg_loss(solution; skip=0)
    return mean(solution.observations[(skip+1):end])
end

function construct_tps_data_loss_vs_s_plot(; new_plot=true, kwargs...)
    s_values, t_values, _, losses = load_processed_tps_data()
    max_s = maximum(s_values)
    min_s = minimum(s_values)
    fig = construct_exact_linear_data_loss_vs_s_plot(false; min_s, max_s, new_plot, kwargs...)

    markers = [:ltriangle, :diamond, :rect, :dtriangle, :circle]
    fig = plot_s_graph(s_values, t_values, losses;
        new_plot=false,
        markershape=markers,
        markersize=3,
        legend_column=2,
        linecolor=nothing,
        kwargs...
    )

    return fig
end

function plot_tps_linear_figure()
    fig = construct_tps_data_loss_vs_s_plot()
    save(joinpath("figures", "tps_linear_perceptron.pdf"), fig)
    return fig
end