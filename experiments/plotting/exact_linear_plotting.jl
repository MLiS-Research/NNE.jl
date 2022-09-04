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

function construct_exact_linear_data_loss_vs_s_plot(plot_inset=true; min_s=nothing, max_s=nothing, inset_max_s=0.1, kwargs...)
    s_values, t_values, _, _, losses = load_exact_linear_data()
    if !isnothing(max_s) || !isnothing(min_s)
        selection_indices = s_values .>= (isnothing(min_s) ? typemin(eltype(s_values)) : min_s)
        selection_indices .&= s_values .<= (isnothing(max_s) ? typemax(eltype(s_values)) : max_s)
        s_values = s_values[selection_indices]
        losses = losses[selection_indices, :]
    end
    plt = begin
        plt = plot_s_graph(s_values, t_values, losses;
            ticks_kwargs=Dict(:round_digits => 0, :power_step => 2),
            linestyle=[:solid :dash :dot :dashdot :dashdotdot],
            kwargs...
        )

        if plot_inset
            construct_exact_linear_data_loss_vs_s_plot(false;
                max_s=inset_max_s,
                legend=false,
                inset_subplots=
                [(1, Plots.bbox(0.1, 0.45, 0.45, 0.45))],
                subplot=2,
                new_plot=false,
                background_color_inside=nothing,
                ticks_kwargs=Dict(:round_digits => 0, :power_step => 1),
                framestyle=:box,
                kwargs...
            )
            ylabel!(plt[2], "")
            xlabel!(plt[2], "")

        end

        return plt
    end

    return plt
end

function plot_exact_linear_figure()
    plt = construct_exact_linear_data_loss_vs_s_plot()
    savefig(plt, "figures/exact_linear_perceptron.pdf")
    return plt
end