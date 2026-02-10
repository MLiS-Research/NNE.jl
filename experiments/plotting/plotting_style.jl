using CairoMakie
using LaTeXStrings
using ColorSchemes


function get_figure_size(; columns=1, dpi=300, height_ratio=1)
    # Calculate size in mm
    target_size = (86 * columns, 86 * columns * height_ratio)
    # Convert mm to points (1 mm = 2.83465 pt)
    mm_to_pt = 2.83465
    size_pt = (target_size[1] * mm_to_pt, target_size[2] * mm_to_pt)

    return size_pt
end

function get_plot_defaults(; columns=1, dpi=300, height_ratio=1)
    fig_size = get_figure_size(; columns, dpi, height_ratio)

    return Dict{Symbol,Any}(
        :size => fig_size,
        :resolution => fig_size,  # For compatibility
        :pt_per_unit => dpi / 72.0,
        :linewidth => 3,
        :markersize => 18,
    )
end

function calculate_axis_ticks(min_v, max_v; base=10, power_step=2, round_digits=0, kwargs...)

    min_exp = round(log(base, min_v), RoundDown) - 1
    max_exp = round(log(base, max_v), RoundUp) + 1

    powers = collect(min_exp:power_step:max_exp)
    rounded_powers = unique(round_digits == 0 ? Int.(powers) : (x -> round(x; digits=round_digits)).(powers))

    labels = (x -> LaTeXString("\$ $base ^ {$x} \$")).(rounded_powers)

    return (Float64(base) .^ rounded_powers, labels)
end


function plot_s_graph(s_values, τ_values, losses; fig=nothing, gridpos=(1, 1), ticks_kwargs=Dict{Symbol,Any}(), show_ylabel=true, show_legend=true, linestyle=nothing, markershape=nothing, use_markers=true, extra_plot_kwargs=Dict{Symbol,Any}(), ylims=nothing, kwargs...)
    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i) / (length(τ_values)) * 256))] for i in 1:length(τ_values)]

    plot_defaults = get_plot_defaults()
    s_ticks = calculate_axis_ticks(minimum(s_values), maximum(s_values); ticks_kwargs...)

    # Use provided ylims or calculate from data
    loss_min = isnothing(ylims) ? minimum(losses) : ylims[1]
    loss_max = isnothing(ylims) ? maximum(losses) : ylims[2]
    loss_ticks = calculate_axis_ticks(loss_min, loss_max; ticks_kwargs...)

    # Create figure if not provided
    if isnothing(fig)
        fig = Figure(; size=plot_defaults[:size])
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

    # Handle linestyle array or single value
    linestyles = isnothing(linestyle) ? fill(:solid, length(τ_values)) : (linestyle isa AbstractArray ? linestyle : fill(linestyle, length(τ_values)))
    markers = isnothing(markershape) ? fill(:circle, length(τ_values)) : (markershape isa AbstractArray ? markershape : fill(markershape, length(τ_values)))

    # Plot lines for each τ value
    for (i, τ) in enumerate(τ_values)
        if !use_markers
            lines!(ax, s_values, losses[:, i],
                color=colors[i],
                linewidth=plot_defaults[:linewidth],
                linestyle=linestyles[i],
                label=LaTeXString("\$\\tau=$τ\$"),
                extra_plot_kwargs...
            )
        else
            scatterlines!(ax, s_values, losses[:, i],
                color=colors[i],
                linewidth=plot_defaults[:linewidth],
                marker=markers[i],
                linestyle=linestyles[i],
                label=LaTeXString("\$\\tau=$τ\$"),
                markersize=plot_defaults[:markersize],
                strokecolor=:black,
                strokewidth=1,
                extra_plot_kwargs...
            )
        end
    end

    # Add legend if requested
    if show_legend
        axislegend(ax, position=:rt, framevisible=false)
    end

    return fig
end