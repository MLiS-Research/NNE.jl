using Plots
using LaTeXStrings
using Measures: mm
# pyplot()



function get_plot_defaults()
    return Dict{Symbol, Any}(
        :thickness_scaling=>2,
        :margin=>1*mm,
        :left_margin=>-12*mm,
        :bottom_margin=>-12*mm,
        :grid=>nothing,
        :lw=>2,
        :size=>(600, 600),
        :legend_background_color=>nothing,
        :legend_foreground_color=>nothing,
        :dpi=>600
    )
end

function plot_s_graph(s_values, τ_values, losses; new_plot=true, kwargs...)
    color_palette = cgrad(:matter)
    colors = [color_palette[Int(round((i)/(length(τ_values))*256))] for i in 1:length(τ_values)]
    labels = reshape(["τ=$t" for t in τ_values], 1, :)
    plot_fn = new_plot ? plot : plot!
    plot_defaults = get_plot_defaults()
    plt = plot_fn(s_values, losses, labels=labels;
     yscale=:log10,
     xscale=:log10,
     legend=:topright,
     palette=colors,
     plot_defaults...,
     kwargs...
    )
    xlabel!(L"s")
    ylabel!(L"\mathbb{E}(\overline{\mathcal{L}})")
    return plt
end

# Larger text size (x2-3)
# Get rid of the box on the legend
# Border on the figure
# Turn off grid lines
# Have different styles for each line