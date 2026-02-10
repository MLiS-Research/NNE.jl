include("plotting_style.jl")
include("plotting_utilities.jl")
using NNE
using CairoMakie
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

function construct_exact_linear_data_loss_vs_s_plot(plot_inset=true; min_s=nothing, max_s=nothing, inset_max_s=0.1, results=load_exact_linear_data(), kwargs...)
    s_values, t_values, _, _, losses = results
    if !isnothing(max_s) || !isnothing(min_s)
        selection_indices = s_values .>= (isnothing(min_s) ? typemin(eltype(s_values)) : min_s)
        selection_indices .&= s_values .<= (isnothing(max_s) ? typemax(eltype(s_values)) : max_s)
        s_values = s_values[selection_indices]
        losses = losses[selection_indices, :]
    end

    fig = plot_s_graph(s_values, t_values, losses;
        ticks_kwargs=Dict(:round_digits => 0, :power_step => 2),
        linestyle=[:solid, :dash, :dot, :dashdot, :dashdotdot],
        kwargs...
    )

    # Note: Inset plotting functionality would need to be implemented separately for CairoMakie
    # For now, we'll skip the inset subplot feature

    return fig
end

function plot_exact_linear_figure()
    fig = construct_exact_linear_data_loss_vs_s_plot()
    save(joinpath("figures", "exact_linear_perceptron.pdf"), fig)
    return fig
end