"""
Functions for creating comprehensive linear problem plots combining exact and TPS results.
"""

include("plotting_utilities.jl")
include("tps_linear_plotting.jl")

using CairoMakie
using LaTeXStrings
using NNE.ExactCalculation
using Base.Iterators

"""
    calculate_exact_linear_losses(s_values, t_values, σ_values; problem=construct_problem())

Calculate losses for given s, t, and σ values using exact calculation.
"""
function calculate_exact_linear_losses(s_values, t_values, σ_values; problem=construct_problem())
    map(x -> get_mean_time_integrated_loss(x..., problem), product(s_values, t_values, σ_values))
end

"""
    plot_comprehensive_linear_problem()

Create a comprehensive figure with three panels showing linear problem results:
- Panel (a): Small sigma (σ=0.1) exact results
- Panel (b): Unity sigma (σ=1.0) exact results
- Panel (c): Empirical TPS data
"""
function plot_comprehensive_linear_problem()
    s_values, t_values, _, _ = load_processed_tps_linear_data()

    min_s = minimum(s_values)
    max_s = maximum(s_values)
    num_s_exact = 1000
    s_values_exact = exp_spaced_values(min_s, max_s, num_s_exact)

    # Calculate losses for both sigma values
    unity_sigma_losses = reshape(calculate_exact_linear_losses(s_values_exact, t_values, [1.0]),
        length(s_values_exact), length(t_values))
    small_sigma_losses = reshape(calculate_exact_linear_losses(s_values_exact, t_values, [0.1]),
        length(s_values_exact), length(t_values))
    tps_losses = load_processed_tps_linear_data()[4]

    # Calculate common y-axis limits across all three datasets with padding for log scale
    global_min_loss = min(minimum(small_sigma_losses), minimum(unity_sigma_losses), minimum(tps_losses))
    global_max_loss = max(maximum(small_sigma_losses), maximum(unity_sigma_losses), maximum(tps_losses))

    # Add multiplicative padding for log scale
    padding_factor = 1.5
    common_ylims = (global_min_loss / padding_factor, global_max_loss * padding_factor)

    # Create a combined figure with 2 panels
    fig = create_publication_figure(num_panels=2, num_panels_y=1)

    # Panel (a): Small sigma
    plot_loss_vs_s_parameter(s_values_exact, t_values, small_sigma_losses;
        fig=fig, gridpos=(1, 1),
        ticks_kwargs=Dict(:round_digits => 0, :power_step => 2),
        linestyle=[(:solid, :dense), (:dash, :dense), (:dot, :dense), (:dashdot, :dense), (:dashdotdot, :dense)],
        show_ylabel=true,
        use_markers=false,
        ylims=common_ylims
    )

    # Panel (b): Unity sigma
    plot_loss_vs_s_parameter(s_values_exact, t_values, unity_sigma_losses;
        fig=fig, gridpos=(1, 2),
        ticks_kwargs=Dict(:round_digits => 0, :power_step => 2),
        linestyle=[(:solid, :dense), (:dash, :dense), (:dot, :dense), (:dashdot, :dense), (:dashdotdot, :dense)],
        show_ylabel=false,
        use_markers=false,
        ylims=common_ylims
    )

    # Panel (c): Empirical TPS data
    markers = [:ltriangle, :diamond, :rect, :dtriangle, :circle]
    plot_loss_vs_s_parameter(s_values, t_values, tps_losses;
        fig=fig, gridpos=(1, 3),
        ticks_kwargs=Dict(:round_digits => 0, :power_step => 2),
        linestyle=[(:solid, :dense), (:dash, :dense), (:dot, :dense), (:dashdot, :dense), (:dashdotdot, :dense)],
        markershape=markers,
        show_ylabel=false,
        ylims=common_ylims
    )

    # Add panel labels
    Label(fig[1, 1, TopLeft()], L"(a)", padding=(5, 5, 5, 5), halign=:left)
    Label(fig[1, 2, TopLeft()], L"(b)", padding=(5, 5, 5, 5), halign=:left)
    Label(fig[1, 3, TopLeft()], L"(c)", padding=(5, 5, 5, 5), halign=:left)

    return fig
end

"""
    save_comprehensive_linear_problem_figure()

Create the comprehensive linear problem figure and save it to disk.
"""
function save_comprehensive_linear_problem_figure()
    fig = plot_comprehensive_linear_problem()
    save(joinpath("figures", "all_linear_plots.pdf"), fig)
    return fig
end
