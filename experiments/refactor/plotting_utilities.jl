"""
Utility functions for creating publication-quality plots.
Merged from previous PlottingStyle and PlottingUtilities modules.
"""

using CairoMakie
using LaTeXStrings
using ColorSchemes
using Measures: cm, mm, inch
using DataFrames

# Default settings
pub_default_dpi() = 144
pub_default_fontsize() = 10

"""
    create_publication_figure(; dpi=pub_default_dpi(), fontsize=pub_default_fontsize(), num_panels=1, num_panels_y=1, kwargs...)

Create a publication-quality figure with specified DPI and font size.
The figure size is based on 8.6cm per panel width and a 21/28 aspect ratio per panel height.
"""
function create_publication_figure(; dpi=pub_default_dpi(), fontsize=pub_default_fontsize(), num_panels=1, num_panels_y=1, kwargs...)
    resolution = Int.(round.((8.6cm * num_panels, 8.6cm * 21 / 28 * num_panels_y) ./ (1inch) .* dpi))
    pt_in_mm = 0.352777777777778mm
    font_height = Int(round(fontsize * pt_in_mm / 1inch * dpi))
    f = Figure(; fontsize=font_height, fonts=(; regular="Computer Modern"), resolution, dpi, kwargs...)
    return f
end

"""
    calculate_log_axis_ticks(min_v, max_v; base=10, power_step=2, round_digits=0)

Calculate axis tick positions and labels for logarithmic scales.
"""
function calculate_log_axis_ticks(min_v, max_v; base=10, power_step=2, round_digits=0)
    min_exp = round(log(base, min_v), RoundDown) - 1
    max_exp = round(log(base, max_v), RoundUp) + 1

    powers = collect(min_exp:power_step:max_exp)
    rounded_powers = unique(round_digits == 0 ? Int.(powers) : (x -> round(x; digits=round_digits)).(powers))

    labels = (x -> LaTeXString("\$ $base ^ {$x} \$")).(rounded_powers)

    return (Float64(base) .^ rounded_powers, labels)
end

"""
    plot_loss_vs_s_parameter(s_values, τ_values, losses; fig=nothing, gridpos=(1, 1), 
                             ticks_kwargs=Dict{Symbol,Any}(), show_ylabel=true, show_legend=true, 
                             linestyle=nothing, markershape=nothing, use_markers=true, 
                             ylims=nothing, kwargs...)

Create a plot of loss values vs s parameter for different τ values with log-log scale.
"""
function plot_loss_vs_s_parameter(s_values, τ_values, losses;
    fig=nothing, gridpos=(1, 1),
    ticks_kwargs=Dict{Symbol,Any}(),
    show_ylabel=true, show_legend=true,
    linestyle=nothing, markershape=nothing,
    use_markers=true, ylims=nothing, kwargs...)
    # Color scheme
    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i) / (length(τ_values)) * 256))] for i in 1:length(τ_values)]

    # Calculate ticks
    s_ticks = calculate_log_axis_ticks(minimum(s_values), maximum(s_values); ticks_kwargs...)
    loss_min = isnothing(ylims) ? minimum(losses) : ylims[1]
    loss_max = isnothing(ylims) ? maximum(losses) : ylims[2]
    loss_ticks = calculate_log_axis_ticks(loss_min, loss_max; ticks_kwargs...)

    # Create figure if not provided
    if isnothing(fig)
        fig = create_publication_figure()
    end

    ax = Axis(fig[gridpos...],
        xlabel=L"s",
        ylabel=show_ylabel ? L"\mathbb{E}[\overline{\mathcal{L}}] / \tau" : "",
        xscale=log10,
        yscale=log10,
        xticks=s_ticks,
        yticks=loss_ticks,
        rightspinevisible=false,
        topspinevisible=false,
        xgridvisible=false,
        ygridvisible=false
    )

    # Set ylims if provided
    if !isnothing(ylims)
        CairoMakie.ylims!(ax, ylims...)
    end

    # Handle linestyle and marker arrays
    linestyles = isnothing(linestyle) ? fill(:solid, length(τ_values)) :
                 (linestyle isa AbstractArray ? linestyle : fill(linestyle, length(τ_values)))
    markers = isnothing(markershape) ? fill(:circle, length(τ_values)) :
              (markershape isa AbstractArray ? markershape : fill(markershape, length(τ_values)))

    # Plot lines for each τ value
    for (i, τ) in enumerate(τ_values)
        if !use_markers
            lines!(ax, s_values, losses[:, i],
                color=colors[i],
                linewidth=3,
                linestyle=linestyles[i],
                label=LaTeXString("\$\\tau=$τ\$")
            )
        else
            scatterlines!(ax, s_values, losses[:, i],
                color=colors[i],
                linewidth=3,
                marker=markers[i],
                linestyle=linestyles[i],
                label=LaTeXString("\$\\tau=$τ\$"),
                markersize=18,
                strokecolor=:black,
                strokewidth=1
            )
        end
    end

    # Add legend if requested
    if show_legend
        axislegend(ax, position=:rt, framevisible=false)
    end

    return fig
end

"""
    get_tau_marker_mapping(tau_values::AbstractArray{Int})

Generate a mapping from tau values to marker shapes for plotting.
"""
function get_tau_marker_mapping(tau_values::AbstractArray{Int})
    possible_markers = [:circle, :diamond, :rect, :utriangle, :star4, :xcross]
    mapping = Dict{Int,Symbol}(
        t => m for (t, m) in Iterators.zip(tau_values, possible_markers)
    )
    return mapping
end

"""
    exp_spaced_values(min_value, max_value, n)

Generate exponentially spaced values between min_value and max_value.
"""
exp_spaced_values(min_value, max_value, n) = exp.((LinRange(log(min_value), log(max_value), n)))
