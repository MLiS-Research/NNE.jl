include("plotting_utilities.jl")
include("tps_linear_plotting.jl")
include("../exact_linear_perceptron.jl")
using CairoMakie
using LaTeXStrings

function plot_all()
    s_values, t_values, _, _ = load_processed_tps_data()

    min_s = minimum(s_values)
    max_s = maximum(s_values)
    num_s_exact = 1000
    s_values_exact = exp_spaced_values(min_s, max_s, num_s_exact)

    # Calculate losses for both sigma values
    unity_sigma_losses = reshape(calc_losses(s_values_exact, t_values, [1.0]), length(s_values_exact), length(t_values))
    small_sigma_losses = reshape(calc_losses(s_values_exact, t_values, [0.1]), length(s_values_exact), length(t_values))
    tps_losses = load_processed_tps_data()[4]

    # Calculate common y-axis limits across all three datasets with padding for log scale
    global_min_loss = min(minimum(small_sigma_losses), minimum(unity_sigma_losses), minimum(tps_losses))
    global_max_loss = max(maximum(small_sigma_losses), maximum(unity_sigma_losses), maximum(tps_losses))

    # Add multiplicative padding for log scale
    padding_factor = 1.5
    common_ylims = (global_min_loss / padding_factor, global_max_loss * padding_factor)

    # Create a combined figure with 3 panels
    fig = create_pub_fig(num_panels=3, num_panels_y=1)

    # Panel (a): Small sigma
    plot_s_graph(s_values_exact, t_values, small_sigma_losses;
        fig=fig, gridpos=(1, 1),
        ticks_kwargs=Dict(:round_digits => 0, :power_step => 2),
        linestyle=[(:solid, :dense), (:dash, :dense), (:dot, :dense), (:dashdot, :dense), (:dashdotdot, :dense)],
        show_ylabel=true,
        use_markers=false,
        ylims=common_ylims
    )

    # Panel (b): Unity sigma
    plot_s_graph(s_values_exact, t_values, unity_sigma_losses;
        fig=fig, gridpos=(1, 2),
        ticks_kwargs=Dict(:round_digits => 0, :power_step => 2),
        linestyle=[(:solid, :dense), (:dash, :dense), (:dot, :dense), (:dashdot, :dense), (:dashdotdot, :dense)],
        show_ylabel=false,
        use_markers=false,
        ylims=common_ylims
    )

    # Panel (c): Empirical TPS data
    markers = [:ltriangle, :diamond, :rect, :dtriangle, :circle]
    plot_s_graph(s_values, t_values, tps_losses;
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

function plot_all_and_save()
    fig = plot_all()
    save(joinpath("figures", "all_linear_plots.pdf"), fig)
end

