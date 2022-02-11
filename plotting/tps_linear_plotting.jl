include("plotting_style.jl")
using NNE
using TPS
using Random
using Plots
using Statistics
using BSON: @load, @save
include("exact_linear_plotting.jl")

function load_tps_linear_data()
    s_values = nothing
    t_values = nothing
    σ = nothing
    results = nothing
    losses = nothing
    @load "results/large/tps_linear_perceptron.bson" s_values t_values σ results losses
    return s_values, t_values, σ, results, losses
end

function process_tps_linear_data_and_save()
    s_values, t_values, σ, results, losses = load_tps_linear_data()
    times_arr = hcat(repeat(t_values', length(s_values)))
    losses = get_avg_loss.(results; skip=0) ./ times_arr

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

function construct_tps_data_loss_vs_s_plot()
    s_values, t_values, _, losses = load_processed_tps_data()
    max_s = maximum(s_values)
    plt = construct_exact_linear_data_loss_vs_s_plot(;max_s=max_s)
    plt = plot_s_graph(s_values, t_values, losses; new_plot=false, markershape=[:utriangle :rect :dtriangle :circle], linecolor=nothing)
    return plt
end

function plot_tps_linear_figure()
    plt = construct_tps_data_loss_vs_s_plot()
    savefig(plt, "figures/tps_linear_perceptron.pdf")
end