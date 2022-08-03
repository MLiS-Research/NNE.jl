include("plotting_style.jl")
using NNE
using TPS
using Random
using Plots
using Statistics
using BSON: @load, @save

function load_tps_toy_data()
    s_values = nothing
    t_values = nothing
    σ = nothing
    results = nothing
    @load "results/large/tpy_toy_classification.bson" s_values t_values σ results
    return s_values, t_values, σ, results
end

function process_tps_toy_data_and_save()
    s_values, t_values, σ, results = load_tps_toy_data()
    times_arr = hcat(repeat(t_values', length(s_values)))
    get_last_n_losses(solution, n) = mean(solution.observations[(end-n):end])
    losses = (x->get_last_n_losses(x, 50000)).(results) ./ times_arr

    @save "results/tpy_toy_classification.bson" s_values t_values σ losses
    nothing
end

function load_processed_tps_toy_data()
    s_values = nothing
    t_values = nothing
    σ = nothing
    losses = nothing
    @load "results/tpy_toy_classification.bson" s_values t_values σ losses
    return s_values, t_values, σ, losses
end

function get_avg_loss(solution; skip=0)
    return mean(solution.observations[(skip+1):end])
end

function construct_tps_toy_classification_data_loss_vs_s_plot()
    s_values, t_values, _, losses = load_processed_tps_toy_data()
    plt = plot_s_graph(s_values, t_values, losses; new_plot=false, markershape=[:utriangle :rect :dtriangle :circle], linecolor=nothing)
    return plt
end

function plot_tps_toy_classification_figure()
    plt = construct_tps_toy_classification_data_loss_vs_s_plot()
    savefig(plt, "figures/tpy_toy_classification.pdf")
end