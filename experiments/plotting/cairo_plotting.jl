using CairoMakie
using LaTeXStrings
using Measures: mm, cm, inch
using DataFrames
using ColorSchemes
import MLDatasets
Makie = CairoMakie

default_dpi() = 144
default_fontsize() = 10
function create_pub_fig(; dpi=default_dpi(), fontsize=default_fontsize(), num_panels=1, num_panels_y=1, kwargs...)
    resolution = Int.(round.((8.6cm * num_panels, 8.6cm * 21 / 28 * num_panels_y) ./ (1inch) .* dpi))
    pt_in_mm = 0.352777777777778mm
    font_height = Int(round(fontsize * pt_in_mm / 1inch * dpi))
    f = Figure(; fontsize=font_height, fonts=(; regular="Computer Modern"), resolution, dpi, kwargs...)
    return f
end
function get_marker_shape_dict(tau_values::AbstractArray{Int})
    possible_markers = [:circle, :diamond, :rect, :utriangle, :start4, :xcross]

    mapping = Dict{Int,Symbol}(
        t => m for (t, m) in Iterators.zip(tau_values, possible_markers)
    )
    return mapping
end
function get_marker_shape_dict(df::DataFrame)
    taus = sort(unique(df[!, :tau]))
    return get_marker_shape_dict(taus)
end
function plot_generic_quantity(df, x_symbol, y_symbol; xlabel=nothing, ylabel=nothing, legend_pos=:rt, legend_nbanks=1, error_whisker_width=20, connect_lines=true, existing_fig=nothing, axis_index=(1, 1), y_modifier=identity, x_modifier=identity, yscale=Makie.Makie.pseudolog10, xscale=Makie.Makie.pseudolog10, axis_kwargs=Dict{Symbol,Any}(), kwargs...)
    tau_values = sort(unique(df.tau))
    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i) / (length(tau_values)) * 256))] for i in 1:length(tau_values)]
    color_map = Dict(tau => c for (tau, c) in zip(tau_values, colors))
    marker_map = get_marker_shape_dict(df)
    fontsize = default_fontsize()
    f = if isnothing(existing_fig)
        create_pub_fig()
    else
        existing_fig
    end

    ax = Axis(f[axis_index...]; yscale, xscale, xlabel, ylabel, subtitlesize=fontsize, ygridvisible=false, xgridvisible=false, axis_kwargs...)

    y_err_symbol = Symbol(:error_, y_symbol)
    x_err_symbol = Symbol(:error_, x_symbol)

    for _df in groupby(df, :tau)
        sort!(_df, [x_symbol])
        tau = first(_df[!, :tau])
        x_values = x_modifier.(_df[!, x_symbol])
        y_values = y_modifier.(_df[!, y_symbol])
        colour = color_map[tau]
        # Error plotting
        if connect_lines
            Makie.lines!(x_values, y_values, linestyle=:dash, label=nothing, color=colour, kwargs...)
        end

        if hasproperty(_df, x_err_symbol)
            x_errors = x_modifier.(_df[!, x_err_symbol])
            Makie.errorbars!(x_values, y_values, x_errors, direction=:x, whiskerwidth=error_whisker_width)
        end
        if hasproperty(_df, y_err_symbol)
            y_errors = y_modifier.(_df[!, y_err_symbol])
            Makie.errorbars!(x_values, y_values, y_errors, direction=:y, whiskerwidth=error_whisker_width)
        end
        Makie.scatter!(x_values, y_values, label=LaTeXString("\$\\tau=$tau \$"), color=colour, marker=marker_map[tau], kwargs...)
    end

    axislegend(ax, position=legend_pos, orientation=:vertical, nbanks=legend_nbanks, framevisible=false, bgcolor=nothing)
    return f
end

function save_fig(root_filename, fig)
    if !isdir("figures")
        mkdir("figures")
    end
    if !isdir("figures/pdf")
        mkdir("figures/pdf")
    end
    if !isdir("figures/svg")
        mkdir("figures/svg")
    end
    Makie.save("figures/pdf/$(root_filename).pdf", fig; pt_per_unit=1)
    Makie.save("figures/svg/$(root_filename).svg", fig; pt_per_unit=1)
end
function plot_s_ensemble(tau_values, analytic_s_values, analytic_losses, empirical_s_values, empircal_losses, loss_subscript=nothing)
    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i) / (length(tau_values)) * 256))] for i in 1:length(tau_values)]
    color_map = Dict(tau => c for (tau, c) in zip(sort(tau_values), colors))
    marker_map = get_marker_shape_dict(tau_values)
    fontsize = default_fontsize()
    f = create_pub_fig()
    ylbl = isnothing(loss_subscript) ? L"\left \langle \mathcal{L} \right \rangle / \tau" : LaTeXString("\$\\left \\langle \\mathcal{L}_{$loss_subscript} \\right \\rangle / \\tau \$")
    ax = Axis(f[1, 1], yscale=log10, xscale=log10, xlabel=L"s", ylabel=ylbl, subtitlesize=fontsize, ygridvisible=false, xgridvisible=false)
    for (j, tau) in enumerate(tau_values)
        colour = color_map[tau]
        Makie.lines!(analytic_s_values, analytic_losses[:, j], label=" ", linestyle=:dash, color=colour)
        Makie.scatter!(empirical_s_values, empircal_losses[:, j], label=LaTeXString("\$ \\tau=$tau \$"), marker=marker_map[tau], color=colour)
    end
    axislegend(ax, position=:rt, orientation=:vertical, nbanks=2, framevisible=false, bgcolor=nothing)
    return f
end

function plot_s_ensemble(df::DataFrame; kwargs...)
    xlabel = L"s"
    ylabel = LaTeXString("\$ \\left \\langle \\mathcal{L}_\\text{MNIST} \\right \\rangle / \\tau \$")
    return plot_generic_quantity(df, :s, :time_avg_loss; xlabel, ylabel, legend_pos=:rt, kwargs...)
end
function plot_accuracies(df::DataFrame, accuracy_sym; kwargs...)
    xlabel = L"s"
    ylabel = LaTeXString("Accuracy (%)")
    y_modifier = x -> x * 100
    yscale = identity
    return plot_generic_quantity(df, :s, accuracy_sym; yscale, xlabel, ylabel, y_modifier, legend_pos=:rb, kwargs...)
end

function plot_avg_batch_sizes(df::DataFrame; kwargs...)
    xlabel = LaTeXString("\$ \\left \\langle \\mathcal{L}_\\text{MNIST} \\right \\rangle / \\tau \$")
    ylabel = L"\left \langle B \right \rangle"
    return plot_generic_quantity(df, :time_avg_loss, :mean_batch_size; xlabel, ylabel, legend_pos=:rt, kwargs...)
end

function plot_avg_cutoff_sizes(df::DataFrame; kwargs...)
    xlabel = LaTeXString("\$ \\left \\langle \\mathcal{L}_\\text{MNIST} \\right \\rangle / \\tau \$")
    ylabel = L"\left \langle c \right \rangle"
    return plot_generic_quantity(df, :time_avg_loss, :mean_cutoff_rate; xlabel, ylabel, legend_pos=:lt, kwargs...)
end

function plot_convergence_autocorrelation(df::DataFrame; kwargs...)
    ylabel = LaTeXString("Autocorrelation Time (Samples)")
    # xlabel = LaTeXString("\$ \\left \\langle L_\\text{MNIST} \\right \\rangle / \\tau \$")
    xlabel = LaTeXString("\$s\$")
    non_missing_df = df[(x->!ismissing(x)).(df.convergence_times), :]
    # return plot_generic_quantity(non_missing_df, :s, :convergence_times; xlabel, ylabel, legend_pos=:rt, yscale=identity, kwargs...)
    return plot_generic_quantity(non_missing_df, :s, :convergence_times; xlabel, ylabel, legend_pos=:rt, kwargs...)
end
function plot_convergence(df::DataFrame; kwargs...)
    ylabel = LaTeXString("Convergence (# Samples)")
    # xlabel = LaTeXString("\$ \\left \\langle L_\\text{MNIST} \\right \\rangle / \\tau \$")
    xlabel = LaTeXString("\$s\$")
    non_missing_df = df[(x->!ismissing(x)).(df.convergence_times), :]
    # return plot_generic_quantity(non_missing_df, :s, :convergence_times; xlabel, ylabel, legend_pos=:rt, yscale=identity, kwargs...)
    return plot_generic_quantity(non_missing_df, :s, :convergence_times; xlabel, ylabel, legend_pos=:lt, yscale=identity, kwargs...)
end

function get_mnist_sample_images(n_digits=10)
    mnist_data = MLDatasets.MNIST(:train)
    digits = Array{Float32,2}[]

    for d in 0:(n_digits-1)
        for (i, true_digit) in enumerate(mnist_data.targets)
            if true_digit == d
                image = reshape(mnist_data.features[:, :, i], 28, 28)
                push!(digits, image)
                break
            end
        end
    end
    return digits
end
function plot_mnist_combined_figure(df::DataFrame, accuracy_sym)
    fig = create_pub_fig(; num_panels=17.8 / 8.6, figure_padding=10)

    # MNIST digits
    mnist_digits = get_mnist_sample_images(10)
    for (i, digit) in enumerate(mnist_digits)
        col = (i - 1) ÷ 5 + 1
        row = (i - 1) % 5 + 1
        ax = Axis(fig[row, col], aspect=1)
        Makie.hidedecorations!(ax)
        Makie.image!(ax, reverse(digit, dims=2))
    end

    axis_kwargs = Dict{Symbol,Any}(
        :xticks => [10, 30, 50]
    )
    plot_s_ensemble(df; existing_fig=fig, axis_index=(1:5, 3), axis_kwargs)
    axis_kwargs = Dict{Symbol,Any}(
        :xticks => [10, 30, 50],
        :yticks => [10, 30, 50, 70, 90]
    )
    plot_accuracies(df, accuracy_sym; existing_fig=fig, axis_index=(1:5, 4), axis_kwargs)
    axis_kwargs = Dict{Symbol,Any}(
        :xticks => [0.5, 1.5, 2.5],
        :yticks => [240, 500, 1000]
    )
    plot_avg_batch_sizes(df; existing_fig=fig, axis_index=(1:5, 5), axis_kwargs)

    for (i, label) in zip((1, 3, 4, 5), ("a", "b", "c", "d"))
        Makie.Label(fig.layout[1, i, Makie.TopLeft()], LaTeXString("($label)"); halign=:right, padding=(0, 5, 10, 0))
    end

    Makie.colgap!(fig.layout, 1, 5)
    Makie.colsize!(fig.layout, 1, Auto(0.15))
    Makie.colsize!(fig.layout, 2, Auto(0.15))

    return fig
end
function plot_combined_losses_figure(individual_run_taus, individual_run_paths; inset_limits_1, inset_labels_1, inset_limits_2, inset_labels_2)
    fig = create_pub_fig(; num_panels=17.8 / 8.6, figure_padding=10)

    # MNIST digits
    mnist_digits = get_mnist_sample_images(10)
    for (i, digit) in enumerate(mnist_digits)
        col = (i - 1) ÷ 5 + 1
        row = (i - 1) % 5 + 1
        ax = Axis(fig[row, col], aspect=1)
        Makie.hidedecorations!(ax)
        Makie.image!(ax, reverse(digit, dims=2))
    end

    axis_kwargs = Dict{Symbol,Any}(
        :xticks => [10, 30, 50]
    )
    plot_s_ensemble(df; existing_fig=fig, axis_index=(1:5, 3), axis_kwargs)
    axis_kwargs = Dict{Symbol,Any}(
        :xticks => [10, 30, 50],
        :yticks => [10, 30, 50, 70, 90]
    )
    plot_accuracies(df, accuracy_sym; existing_fig=fig, axis_index=(1:5, 4), axis_kwargs)
    axis_kwargs = Dict{Symbol,Any}(
        :xticks => [0.5, 1.5, 2.5],
        :yticks => [240, 500, 1000]
    )
    plot_avg_batch_sizes(df; existing_fig=fig, axis_index=(1:5, 5), axis_kwargs)

    for (i, label) in zip((1, 3, 4, 5), ("a", "b", "c", "d"))
        Makie.Label(fig.layout[1, i, Makie.TopLeft()], LaTeXString("($label)"); halign=:right, padding=(0, 5, 10, 0))
    end

    Makie.colgap!(fig.layout, 1, 5)
    Makie.colsize!(fig.layout, 1, Auto(0.15))
    Makie.colsize!(fig.layout, 2, Auto(0.15))

    return fig
end

to_instant(d) = d.instant.periods.value
function time_block(start_time, end_time, index)
    return Rect((start_time + end_time) / 2, index + 0.5, end_time - start_time, 1)
end
function get_colour_scheme(df, key; ascending=true, scheme=:inferno)
    unique_keys = sort(unique(df[!, key]), rev=!ascending)
    cscheme = cgrad(scheme, length(unique_keys); categorical=true)
    return Dict(
        (k => cscheme[i] for (i, k) in enumerate(unique_keys))...
    )
end
function plot_occupancy_time(df)
    fontsize = default_fontsize()
    f = create_pub_fig()
    ax = Axis(f[1, 1], xlabel="Time (hrs)", ylabel="Node Index", subtitlesize=fontsize, ygridvisible=false, xgridvisible=false)
    cscheme = get_colour_scheme(df, :tau)
    first_start = minimum(df[!, :start_datetime])
    for _df in groupby(df, :tau)
        tau = first(_df[!, :tau])
        label = "τ=$tau"
        rectangles = map(eachrow(_df)) do row
            start_time = (x -> convert(Dates.Millisecond, x).value).(row[:start_datetime] - first_start) / (1000 * 3600)
            end_time = (x -> convert(Dates.Millisecond, x).value).(row[:end_datetime] - first_start) / (1000 * 3600)
            duration = start_time - end_time
            time_block(start_time, end_time, row[:node_index])
        end
        poly!(rectangles, label=label, color=cscheme[tau])
    end
    axislegend(ax, orientation=:vertical, nbanks=1, framevisible=false, bgcolor=nothing)
    return f
end

# Makie
# function Makie.barplot!(histogram::MinibatchTPS.Histograms.FixedWidthHistogram{IgnoreExtrema}; kwargs...) where {IgnoreExtrema}
#     bin_width = (histogram.bin_edges[begin+1] - histogram.bin_edges[begin])
#     bin_centers = collect(histogram.bin_edges[begin:end-1]) .+ bin_width / 2
#     offset = IgnoreExtrema ? 0 : 1
#     bin_heights = histogram.values[begin+offset:end-offset] ./ histogram.total_entries
#     barplot!(bin_centers, bin_heights; gap=0, width=bin_width, kwargs...)
# end

function loss_curve_plot_v0(file_path, label=nothing, existing_plot=nothing; use_epochs=false, log_x=true, scale_x_by_tau=true, n_points=5000, kwargs...)
    configuration = nothing
    results = nothing
    BSON.@load file_path configuration results

    tau = configuration[:tau]
    losses = (results[:losses] ./ tau)
    spacing = (length(losses) - 1) ÷ n_points
    indices = if log_x
        sort(unique(Int.(round.(10 .^ LinRange(0, log10(length(losses) - 1), n_points)))))
    else
        1:spacing:(length(losses)-1)
    end
    losses = @views losses[indices.+1]
    if !use_epochs
        batch_sizes = results[:minibatch_statistics][:batch_sizes]
        cum_batch_size = cumsum(batch_sizes)
        x = @views cum_batch_size[indices]
    else
        x = indices
    end
    if scale_x_by_tau
        x = x ./ tau
    end
    existing_plot = isnothing(existing_plot) ? Plots.Plot() : existing_plot
    xscale = log_x ? (:log10) : (:identity)
    plt = Plots.plot!(existing_plot, x, losses; label, xscale, yscale=:log10, kwargs...)
    Plots.ylabel!(plt, "<L>/τ")

    xlbl = if !use_epochs
        scale_x_by_tau ? "# Samples / τ" : "# Samples"
    else
        scale_x_by_tau ? "Epochs / τ" : "Epochs"
    end
    Plots.xlabel!(plt, xlbl)
    Plots.xlims!(plt, extrema(x))
    return plt
end

function loss_curve_plot(file_path, label=nothing, existing_plot=nothing; use_epochs=false, log_x=true, scale_x_by_tau=true, n_points=5000, axis_bbox=nothing, axis_kwargs=Dict{Symbol,Any}(), kwargs...)
    configuration = nothing
    results = nothing
    BSON.@load file_path configuration results

    tau = configuration[:tau]
    losses = (results[:losses] ./ tau)
    spacing = (length(losses) - 1) ÷ n_points
    indices = if log_x
        sort(unique(Int.(round.(10 .^ LinRange(0, log10(length(losses) - 1), n_points)))))
    else
        (1:spacing:(length(losses)-1))
    end
    losses = @views losses[indices.+1]
    if !use_epochs
        batch_sizes = results[:minibatch_statistics][:batch_sizes]
        cum_batch_size = cumsum(batch_sizes)
        x = @views cum_batch_size[indices]
    else
        x = indices
    end
    if scale_x_by_tau
        x = x ./ tau
    end
    xlbl = if !use_epochs
        scale_x_by_tau ? L"D / \tau" : L"D"
    else
        scale_x_by_tau ? LaTeXString("\${E} / {\\tau}\$") : L"E"
    end
    ylbl = LaTeXString("\$ \\left \\langle \\mathcal{L}_\\text{MNIST} \\right \\rangle / \\tau \$")
    xscale = log_x ? log10 : identity
    f, ax = if isnothing(existing_plot)
        f = create_pub_fig()
        (f, nothing)
    else
        existing_plot
    end

    if isnothing(ax)
        ax = if isnothing(axis_bbox)
            ax = f[1, 1] = Axis(f; xlabel=xlbl, ylabel=ylbl, xscale, yscale=Makie.Makie.pseudolog10, ygridvisible=false, xgridvisible=false, axis_kwargs...)
            ax
        else
            Axis(f; bbox=axis_bbox, xlabel=xlbl, ylabel=ylbl, xscale, yscale=Makie.Makie.pseudolog10, ygridvisible=false, xgridvisible=false, axis_kwargs...)
        end
    end

    x = collect(Float32, x)
    losses = Float32.(losses)
    Makie.lines!(ax, x, losses; label, linewidth=2, kwargs...)

    Makie.xlims!(ax, extrema(x))

    return (f, ax)
end
function combined_loss_curve_plot(taus, paths; use_epochs=true, inset_limits=nothing, inset_labels=nothing, plt=nothing)
    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i) / (4) * 256))] for i in 1:4]


    for (tau, path, color) in zip(taus, paths, colors)
        plt = loss_curve_plot(path, LaTeXString("\$\\tau=$tau\$"), plt; color, use_epochs, scale_x_by_tau=true)
    end
    (f, ax) = plt
    axislegend(ax, position=:rt, orientation=:vertical, nbanks=1, framevisible=false, bgcolor=nothing)
    dpi = default_dpi()
    original_resolution = Int.(round.((8.6cm, 8.6cm * 21 / 28) ./ (1inch) .* dpi))

    bbox = Makie.BBox(original_resolution[1] / 4, 3 * original_resolution[1] / 5, 4 * original_resolution[2] / 10, 8.5 * original_resolution[2] / 10)
    plt = (f, nothing)
    for (path, color) in zip(paths, colors)
        axis_kwargs = Dict{Symbol,Any}()
        if !isnothing(inset_labels)
            axis_kwargs[:xticks] = inset_labels
        end
        plt = loss_curve_plot(path, "", plt; color, use_epochs, scale_x_by_tau=false, axis_bbox=bbox, log_x=false, axis_kwargs)
    end
    (f, inset_ax) = plt

    if !isnothing(inset_limits)
        Makie.xlims!(inset_ax, inset_limits)
    end

    Makie.hideydecorations!(inset_ax, label=true, ticklabels=true)

    return f
end

function plot_all_loss_curves_v0(file_paths)
    plt = loss_curve_plot_v0(file_paths[begin], "1")
    for (i, path) in enumerate(file_paths[2:end])
        plt = loss_curve_plot_v0(path, string(i + 1), plt)
    end
    return plt
end