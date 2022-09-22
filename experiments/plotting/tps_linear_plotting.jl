include("plotting_style.jl")
using NNE
using TransitionPathSampling
using Random
using Plots
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
    plt = construct_exact_linear_data_loss_vs_s_plot(false; min_s, max_s, lw=2, new_plot, kwargs...)
    markers = [:ltriangle :diamond :rect :dtriangle :circle]
    plt = plot_s_graph(s_values, t_values, losses;
        new_plot=false,
        markershape=markers,
        markersize=3,
        legend_column=2,
        linecolor=nothing,
        kwargs...
    )
    n_rows = length(t_values)
    n_cols = 2
    for i in 1:length(t_values)
        plt.series_list[i].plotattributes[:label] = L""
    end
    plt.series_list = [plt.series_list[1+(i÷n_cols%n_rows)+(i%n_cols)*n_rows] for i in 0:length(plt.series_list)-1]
    for i in 1:length(plt.series_list)

        plt.series_list[i].plotattributes[:series_index] = i
        plt.series_list[i].plotattributes[:series_plotindex] = i
    end
    plt.subplots[1].series_list = plt.series_list

    return plt
end

function plot_tps_linear_figure()
    plt = construct_tps_data_loss_vs_s_plot()
    savefig(plt, "figures/tps_linear_perceptron.pdf")
    return plt
end