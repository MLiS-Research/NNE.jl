using Plots
using LaTeXStrings

function plot_s_graph(s_values, τ_values, losses; kwargs...)
    color_palette = cgrad(:matter)
    colors = [color_palette[Int(round((i)/(length(τ_values))*256))] for i in 1:length(τ_values)]
    labels = reshape(["τ=$t" for t in τ_values], 1, :)
    plt = plot(s_values, losses, labels=labels; yscale=:log10, xscale=:log10, legend=:topright, lw=2, palette=colors, size=(600, 400), kwargs...)
    xlabel!(L"s")
    ylabel!(L"\mathbb{E}(\overline{\mathcal{L}})")
    return plt
end