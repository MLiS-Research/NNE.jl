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

function construct_exact_linear_data_loss_vs_s_plot()
    s_values, t_values, _, _, losses = load_exact_linear_data()
    plt = plot_s_graph(s_values, t_values, losses; linestyle=:dash)
    return plt
end

plt = construct_exact_linear_data_loss_vs_s_plot()
savefig(plt, "figures/exact_linear_perceptron.pdf")