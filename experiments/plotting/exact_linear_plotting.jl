include("plotting_style.jl")
using NNE
using Plots
using BSON: @load

function load_exact_linear_data()
    s_values = nothing
    t_values = nothing
    sigma = nothing
    problem = nothing
    losses = nothing
    @load "results/exact_linear_perceptron.bson" s_values t_values sigma problem losses
    return s_values, t_values, sigma, problem, losses
end

function construct_exact_linear_data_loss_vs_s_plot(;max_s=nothing)
    s_values, t_values, _, _, losses = load_exact_linear_data()
    if max_s !== nothing
        selection_indices = s_values.<=max_s
        s_values = s_values[selection_indices]
        losses = losses[selection_indices, :]
    end
    plt = plot_s_graph(s_values, t_values, losses; linestyle=:dash)
    return plt
end

function plot_exact_linear_figure()
    plt = construct_exact_linear_data_loss_vs_s_plot()
    savefig(plt, "figures/exact_linear_perceptron.pdf")
    return plt
end