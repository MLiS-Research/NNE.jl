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
    return Dict{Symbol, Any}(
        :dpi=>300,
        :thickness_scaling=>2,
        :size=>(600, 600),
        :lw=>2,
        :legend_background_color=>nothing,
        :legend_foreground_color=>nothing,
        :grid=>nothing
    )
end

function get_plot_defaults_full_width()
    fontsize = 12
    font = Plots.font("times", fontsize)
    size_inches = (3.38*2, 3.38)
    dpi = 300
    size_in_px = size_inches .* dpi

    return Dict{Symbol, Any}(
        :thickness_scaling=>3,
        :titlefont=>font,
        :legendfontsize=>fontsize,
        :guidefontsize=>fontsize,
        :tickfontsize=>fontsize,
        :markerstrokewidth=>0.5,
        :markersize=>5,
        :lw => 2,
        :grid=>nothing,
        :size=>size_in_px,
        :legend_background_color=>nothing,
        :legend_foreground_color=>nothing,
        :dpi=>dpi
    )
end

function calculate_axis_ticks(min_v, max_v; base=10, power_step=1, round_digits=1, kwargs...)
    min_exp = round(log(base, min_v), RoundDown) + 1
    max_exp = round(log(base, max_v), RoundUp) - 1

    powers = collect(min_exp:power_step:max_exp)
    # labels = (x->LaTeXString("\$\$ $base ^ $x \$\$")).(powers)
    labels = (x->LaTeXString("\$ $base ^ {$x)} \$")).(powers)
    
    return (base .^ powers, labels)
end


function plot_s_graph(s_values, τ_values, losses; new_plot=true, kwargs...)
    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i)/(length(τ_values))*256))] for i in 1:length(τ_values)]
    labels = reshape(["τ=$t" for t in τ_values], 1, :)
    plot_fn = new_plot ? plot : plot!
    plot_defaults = get_plot_defaults()
    s_ticks = calculate_axis_ticks(minimum(s_values), maximum(s_values); kwargs...)
    loss_ticks = calculate_axis_ticks(minimum(losses), maximum(losses); kwargs...)
    plt = plot_fn(s_values, losses, labels=labels;
     yscale=:log10,
     xscale=:log10,
     color_palette=colors,
     legend_position=:topright,
    #  xticks=s_ticks,
    #  yticks=loss_ticks,
     plot_defaults...,
     kwargs...
    )
    xlabel!(L"s")
    ylabel!(L"\mathbb{E} \left [ \ \overline{\mathcal{L}} \ \right ] / \tau")
    return plt
end

# Larger text size (x2-3)
# Get rid of the box on the legend
# Border on the figure
# Turn off grid lines
# Have different styles for each line