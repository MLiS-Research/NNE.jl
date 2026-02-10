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
        :linewidth => 2,
        :markersize => 12,
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


function plot_s_graph(s_values, τ_values, losses; new_plot=true, ticks_kwargs=Dict{Symbol,Any}(), kwargs...)
    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i) / (length(τ_values)) * 256))] for i in 1:length(τ_values)]

    plot_defaults = get_plot_defaults()
    s_ticks = calculate_axis_ticks(minimum(s_values), maximum(s_values); ticks_kwargs...)
    loss_ticks = calculate_axis_ticks(minimum(losses), maximum(losses); ticks_kwargs...)

    # Create figure and axis
    fig = Figure(; size=plot_defaults[:size])
    ax = Axis(fig[1, 1],
        xlabel=L"s",
        ylabel=L"\mathbb{E}[\overline{\mathcal{L}}] / \tau",
        xscale=log10,
        yscale=log10,
        xticks=s_ticks,
        yticks=loss_ticks
    )

    # Plot lines for each τ value
    for (i, τ) in enumerate(τ_values)
        lines!(ax, s_values, losses[:, i],
            color=colors[i],
            linewidth=plot_defaults[:linewidth],
            label="τ=$τ"
        )
        scatter!(ax, s_values, losses[:, i],
            color=colors[i],
            markersize=plot_defaults[:markersize]
        )
    end

    # Add legend (top-right, no frame)
    axislegend(ax, position=:rt, framevisible=false)

    return fig
end