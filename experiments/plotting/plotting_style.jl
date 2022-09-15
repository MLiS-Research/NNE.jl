using Plots
pgfplotsx()
using PGFPlotsX
using LaTeXStrings
using Measures: mm
using ColorSchemes


function ensure_pgfplots_packages()
    pgfplots_packages = ["amsmath", "amsfonts"]
    for pkg in pgfplots_packages
        ltx = "\\usepackage{$pkg}"
        if ltx in PGFPlotsX.CUSTOM_PREAMBLE
            continue
        end
        push!(PGFPlotsX.CUSTOM_PREAMBLE, ltx)
    end
end
ensure_pgfplots_packages()

function get_plot_defaults()
    return Dict{Symbol,Any}(
        :dpi => 300,
        :thickness_scaling => 2,
        :size => (400, 400),
        :lw => 2,
        :legend_background_color => nothing,
        :legend_foreground_color => nothing,
        :grid => nothing
    )
end

function get_plot_defaults_full_width()

    return Dict{Symbol,Any}(
        :thickness_scaling => 2,
        :lw => 2,
        :grid => nothing,
        :size => (800, 300),
        :legend_background_color => nothing,
        :legend_foreground_color => nothing,
        :titlefontsize => 10,
        :labelfontsize => 8,
        :dpi => 300
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
    labels = reshape(["τ=$t" for t in τ_values], 1, :)
    plot_fn = new_plot ? plot : plot!
    plot_defaults = get_plot_defaults()
    s_ticks = calculate_axis_ticks(minimum(s_values), maximum(s_values); ticks_kwargs...)
    loss_ticks = calculate_axis_ticks(minimum(losses), maximum(losses); ticks_kwargs...)
    plt = plot_fn(s_values, losses, labels=labels;
        yscale=:log10,
        xscale=:log10,
        color_palette=colors,
        legend_position=:topright,
        xticks=s_ticks,
        yticks=loss_ticks,
        plot_defaults...,
        kwargs...
    )
    xlabel!(plt, L"s")
    ylabel!(plt, L"\mathbb{E} \left [ \ \overline{\mathcal{L}} \ \right ] / \tau")
    return plt
end

# Larger text size (x2-3)
# Get rid of the box on the legend
# Border on the figure
# Turn off grid lines
# Have different styles for each line