using MLDatasets
using CairoMakie
using CairoMakie: Figure, Axis, GridLayout, Label, Legend, DataAspect, Relative
using Images
using Flux
using Statistics
using DataFrames
using NNE.MNISTTraining
using Experimenter
using Base.Iterators
using ProgressBars
using BSON: @save, @load
include("plotting_style.jl")
include("cairo_plotting.jl")

function get_examples(digits...)
    labels = MNIST.trainlabels(1:100)
    indices = [findlast(labels .== d) for d in digits]
    data = MNIST.traintensor(Float32, indices)
    image_data = [(reshape(data[:, :, i], 28, 28)') for i = 1:length(indices)]
    return image_data
end

function create_image_plots(digits...; kwargs...)
    defaults = get_plot_defaults(; columns=2 / 5, height_ratio=5 / 2)

    images = get_examples(digits...)
    n = length(images)

    # Create figure with grid layout
    fig = Figure(; size=defaults[:size])

    # Create 5x2 grid of images
    for (idx, img) in enumerate(images)
        row = (idx - 1) ÷ 2 + 1
        col = (idx - 1) % 2 + 1
        ax = Axis(fig[row, col], aspect=DataAspect())
        heatmap!(ax, rotr90(img), colormap=:grays)
        hidedecorations!(ax)
        hidespines!(ax)
    end

    return fig
end

function create_and_save_mnist_digits()
    fig = create_image_plots((0:9)...)
    save("figures/mnist_digits.pdf", fig)
end

function reconstruct_mnist_models(info_dict; outputs=2, device=cpu)
    model = generate_mnist_model(; outputs)
    _, re = Flux.destructure(model)
    if info_dict[:τ] == 1
        model = re(info_dict[:final_state]) # reconstruct model from params
        return [model |> device]
    end

    models = [re(state) |> device for state in info_dict[:final_state]]
    return models
end

function measure_train_accuracy(info_dict; device=cpu, outputs=2)
    models = reconstruct_mnist_models(info_dict; outputs, device)
    features = info_dict[:features] |> device
    labels = info_dict[:labels] |> cpu
    accuracies = [Flux.mean(reshape((x -> x[1] - 1).(argmax(m(features) |> cpu, dims=1)), :) .== labels) for m in models]
    return accuracies
end

function measure_test_accuracy(info_dict; device=cpu, outputs=2)
    models = reconstruct_mnist_models(info_dict; outputs, device)
    features, labels = get_mnist_testing_dataset(; device, outputs)
    labels = labels |> cpu
    accuracies = [Flux.mean(reshape((x -> x[1] - 1).(argmax(m(features) |> cpu, dims=1)), :) .== labels) for m in models]
    return accuracies
end

function plot_s_vs_loss(trials::AbstractArray{Trial}; kwargs...)
    df = DataFrame(trials)
    prepare_trials_df!(df)
    plot_s_vs_loss(df; kwargs...)
end
function plot_s_vs_loss(df::DataFrame; max_loss_samples=typemax(Int), kwargs...)
    trajectory_lengths = sort(collect(Set(df.τ)))
    defaults = get_plot_defaults()

    fig = Figure(; size=defaults[:size])
    ax = Axis(fig[1, 1], xlabel="s", ylabel="<L>/τ", xscale=log10, yscale=log10)

    marker_shapes = [:circle, :rect, :dtriangle, :utriangle, :diamond, :pentagon]
    colors = Makie.wong_colors()

    for (i, t) in enumerate(trajectory_lengths)
        sub_df = df[df.τ.==t, :]
        s_vals = sort(collect(Set(sub_df.s)))
        losses = Float64[]
        errors = Float64[]
        for s in s_vals
            repeat_ls = (x -> length(x) > max_loss_samples ? x[end-max_loss_samples+1:end] : x).(sub_df[sub_df.s.==s, :losses])
            num_repeats = length(repeat_ls)
            push!(losses, mean(mean(ls) for ls in repeat_ls) / t)
            total_error = std(mean(ls) for ls in repeat_ls) / t / sqrt(num_repeats)
            push!(errors, total_error)
        end
        errorbars!(ax, s_vals, losses, errors, color=colors[mod1(i, length(colors))])
        scatter!(ax, s_vals, losses, label="τ=$t",
            marker=marker_shapes[mod1(i, length(marker_shapes))],
            markersize=defaults[:markersize],
            color=colors[mod1(i, length(colors))])
    end

    axislegend(ax, position=:lb, framevisible=false)
    return fig
end
function plot_accuracy_vs_loss(trials::AbstractArray{Trial}; device=gpu, outputs=10, kwargs...)
    df = DataFrame(trials)
    prepare_trials_df!(df)
    trajectory_lengths = sort(collect(Set(df.τ)))
    defaults = get_plot_defaults()

    fig = Figure(; size=defaults[:size])
    ax = Axis(fig[1, 1], xlabel="s", ylabel="Train Accuracy (%)",
        xscale=log10, yscale=log10, limits=(nothing, nothing, 8, 100))

    marker_shapes = [:circle, :rect, :dtriangle, :utriangle, :diamond, :pentagon]
    colors = Makie.wong_colors()

    for (i, t) in enumerate(trajectory_lengths)
        sub_df = df[df.τ.==t, :]
        s_vals = sort(collect(Set(sub_df.s)))
        accuracies = Float64[]
        errors = Float64[]
        for s in s_vals
            results_list = [x.results for x in trials if x.results[:τ] == t && x.results[:s] == s]
            repeat_accs = measure_train_accuracy.(results_list; device, outputs)
            num_repeats = length(repeat_accs)
            push!(accuracies, mean(mean(as) for as in repeat_accs))
            push!(errors, std(mean(as) for as in repeat_accs) / sqrt(num_repeats))
        end
        errorbars!(ax, s_vals, accuracies .* 100, errors .* 100, color=colors[mod1(i, length(colors))])
        scatter!(ax, s_vals, accuracies .* 100, label="τ=$t",
            marker=marker_shapes[mod1(i, length(marker_shapes))],
            markersize=defaults[:markersize],
            color=colors[mod1(i, length(colors))])
    end

    axislegend(ax, position=:lt, framevisible=false)
    return fig
end
function plot_s_vs_acceptance(trials::AbstractArray{Trial}; kwargs...)
    df = DataFrame(trials)
    prepare_trials_df!(df)
    plot_s_vs_acceptance(df; kwargs...)
end
function plot_s_vs_acceptance(df::DataFrame; max_acceptance_samples=typemax(Int), kwargs...)
    trajectory_lengths = sort(collect(Set(df.τ)))
    defaults = get_plot_defaults()

    fig = Figure(; size=defaults[:size])
    ax = Axis(fig[1, 1], xlabel="s", ylabel="<A>", xscale=log10, yscale=log10)

    marker_shapes = [:circle, :rect, :dtriangle, :utriangle, :diamond]
    colors = Makie.wong_colors()

    for (i, t) in enumerate(trajectory_lengths)
        sub_df = df[df.τ.==t, :]
        s_vals = sort(collect(Set(sub_df.s)))
        acceptances = Float64[]
        errors = Float64[]
        for s in s_vals
            repeat_accepts = (x -> length(x) > max_acceptance_samples ? x[end-max_acceptance_samples+1:end] : x).(sub_df[sub_df.s.==s, :acceptances])
            num_repeats = length(repeat_accepts)
            push!(acceptances, mean(mean(as) for as in repeat_accepts))
            total_error = std(mean(as) for as in repeat_accepts) / sqrt(num_repeats)
            push!(errors, total_error)
        end
        errorbars!(ax, s_vals, acceptances, errors, color=colors[mod1(i, length(colors))])
        scatter!(ax, s_vals, acceptances, label="τ=$t",
            marker=marker_shapes[mod1(i, length(marker_shapes))],
            markersize=defaults[:markersize],
            color=colors[mod1(i, length(colors))])
    end

    axislegend(ax, framevisible=false)
    return fig
end
function prepare_trials_df!(trials_df::DataFrame)
    insertcols!(trials_df, :losses => (x -> x[:observations]).(trials_df.results))
    insertcols!(trials_df, :acceptances => (x -> Float64.(diff(x[:observations]) .== 0)).(trials_df.results))
    insertcols!(trials_df, :s => (x -> x[:s]).(trials_df.configuration))
    insertcols!(trials_df, :τ => (x -> x[:τ]).(trials_df.configuration))
    insertcols!(trials_df, :σ => (x -> x[:σ]).(trials_df.configuration))
end

function plot_avg_loss(results, new_plot=true; should_scale_x=false, fig=nothing, ax=nothing, kwargs...)
    max_len = minimum([length(x[:observations]) for x in results])
    losses = (x -> x[:observations][1:max_len]).(results)
    med_duration = median((x -> x[:duration].value).(results)) ./ 1000.0
    tau = mean((x -> x[:τ]).(results))
    mean_loss = mean(losses) / tau
    std_loss = std(losses) / sqrt(length(losses)) / tau
    x_scale = should_scale_x ? LinRange(0, med_duration, length(mean_loss)) : collect(1:length(mean_loss))

    if new_plot
        defaults = get_plot_defaults()
        fig = Figure(; size=defaults[:size])
        ax = Axis(fig[1, 1],
            xlabel=should_scale_x ? "Runtime (s)" : "Epochs",
            ylabel="Mean Loss")
    end

    if length(mean_loss) > 5e5
        mean_loss = conv_1d(mean_loss, 100, 100.0)
        std_loss = conv_1d(std_loss, 100, 100.0)
        @views band!(ax, x_scale[begin:1000:end],
            (mean_loss.-std_loss)[begin:1000:end],
            (mean_loss.+std_loss)[begin:1000:end], alpha=0.3)
        @views lines!(ax, x_scale[begin:1000:end], mean_loss[begin:1000:end])
    else
        band!(ax, x_scale, mean_loss .- std_loss, mean_loss .+ std_loss, alpha=0.3)
        lines!(ax, x_scale, mean_loss)
    end

    return new_plot ? fig : (fig, ax)
end

function plot_avg_loss_compared(trials; max_y_lim=nothing, kwargs...)
    ts = sort(collect(Set(x.configuration[:τ] for x in trials)))
    ss = sort(collect(Set(x.configuration[:s] for x in trials)))
    split_trials = [[tr for tr in trials if tr.configuration[:τ] == t && tr.configuration[:s] == s] for (s, t) in product(ss, ts)]

    fig = Figure(; size=(600, 1200))

    for (j, s) in enumerate(ss)
        ax = Axis(fig[j, 1],
            xlabel=(j == length(ss) ? "Epochs" : ""),
            ylabel="Mean Loss",
            title="($(Char(96+j)))",
            titlelocation=:left)

        colors = Makie.wong_colors()
        for (i, t) in enumerate(ts)
            results_for_trial = [trial.results for trial in split_trials[j, i]]
            if !isempty(results_for_trial)
                max_len = minimum([length(x[:observations]) for x in results_for_trial])
                losses = (x -> x[:observations][1:max_len]).(results_for_trial)
                tau = mean((x -> x[:τ]).(results_for_trial))
                mean_loss = mean(losses) / tau
                std_loss = std(losses) / sqrt(length(losses)) / tau
                x_scale = collect(1:length(mean_loss))

                band!(ax, x_scale, mean_loss .- std_loss, mean_loss .+ std_loss,
                    alpha=0.3, color=colors[mod1(i, length(colors))])
                lines!(ax, x_scale, mean_loss, label="τ=$t",
                    color=colors[mod1(i, length(colors))])
            end
        end

        if !isnothing(max_y_lim)
            ylims!(ax, 0, max_y_lim)
        end

        if j == 1
            axislegend(ax, position=:rt, framevisible=false)
        end
    end

    return fig
end

function conv_1d(y, w=500, sigma=100.0)
    f(x) = exp(-0.5 * x * x / (sigma * sigma)) / (sigma * sqrt(2 * π))
    kernel = f.(collect(-w:w))
    kernel = kernel ./ sum(kernel)
    conv_y = similar(y)
    for i in eachindex(y)
        min_i = max(1, i - w)
        max_i = min(length(y), i + w)
        kernel_r = (w-(i-min_i)+1):(w+(max_i-i)+1)
        r = (min_i:max_i)
        norm_kernel = kernel[kernel_r]
        norm_kernel ./= sum(norm_kernel)
        conv_y[r] .= sum(y[r] .* norm_kernel)
    end
    return conv_y
end

function plot_acceptance(results, new_plot=true; should_scale_x=false, fig=nothing, ax=nothing, kwargs...)
    acceptances = (x -> Float64.(x[:acceptances])).(results)
    conv_acceptances = (x -> conv_1d(x)).(acceptances)
    med_duration = median((x -> x[:duration].value).(results)) ./ 1000.0
    mean_acceptances = mean(conv_acceptances)
    std_acceptances = std(conv_acceptances)
    x_scale = should_scale_x ? LinRange(0, med_duration, length(mean_acceptances)) : collect(1:length(mean_acceptances))

    if new_plot
        defaults = get_plot_defaults()
        fig = Figure(; size=defaults[:size])
        ax = Axis(fig[1, 1],
            xlabel=should_scale_x ? "Runtime (s)" : "Epochs",
            ylabel="Mean Acceptance")
    end

    band!(ax, x_scale, mean_acceptances .- std_acceptances, mean_acceptances .+ std_acceptances, alpha=0.3)
    lines!(ax, x_scale, mean_acceptances)

    return new_plot ? fig : (fig, ax)
end


function prepare_mnist_results(trials::AbstractArray{Trial}; max_loss_samples=typemax(Int), device=gpu, outputs=10, use_progress=false, kwargs...)
    trajectory_lengths = sort(collect(Set([x.configuration[:τ] for x in trials])))

    results = Dict{Symbol,Any}()
    results[:trajectory_lengths] = trajectory_lengths
    results[:data] = Dict{Int,Any}()

    for (i, t) in enumerate(trajectory_lengths)
        s_vals = sort(collect(Set([x.configuration[:s] for x in trials if x.configuration[:τ] == t])))


        t_data = Dict{Float64,Any}()
        iter = use_progress ? ProgressBar(enumerate(s_vals)) : enumerate(s_vals)
        for (j, s) in iter
            s_data = Dict{Symbol,Any}()
            loss_arrays = [x.results[:observations] for x in trials if x.configuration[:τ] == t && x.configuration[:s] == s]
            repeat_ls = (x -> length(x) > max_loss_samples ? x[end-max_loss_samples+1:end] : x).(loss_arrays)
            num_repeats = length(repeat_ls)
            s_data[:loss] = mean(mean(ls) for ls in repeat_ls) / t
            errors_repeats = [std(ls) / sqrt(length(ls)) for ls in loss_arrays]
            total_error = sqrt(sum(x -> x * x, errors_repeats)) / num_repeats / t
            s_data[:loss_error] = total_error


            results_list = [x.results for x in trials if x.results[:τ] == t && x.results[:s] == s]
            repeat_accs = measure_train_accuracy.(results_list; device, outputs)
            num_repeats = length(repeat_accs)
            s_data[:accuracy] = mean(mean(as) for as in repeat_accs)
            s_data[:accuracy_error] = std(mean(as) for as in repeat_accs) / sqrt(num_repeats)
            s_data[:s] = s
            t_data[s] = s_data
        end
        results[:data][t] = t_data
    end

    @save get_mnist_results_save_path() results
    nothing
end

get_mnist_results_save_path() = joinpath("results", "mnist_data.bson")
get_mnist_results_figure_s_save_path() = joinpath("figures", "full_mnist_s_ensemble.pdf")
get_mnist_results_figure_accuracy_save_path() = joinpath("figures", "full_mnist_accuracy.pdf")
get_combined_mnist_graph_save_path() = joinpath("figures", "all_mnist_graphs.pdf")

function get_mnist_results()
    results = nothing
    @load get_mnist_results_save_path() results
    return results
end

function plot_mnist_s_graph()
    results = get_mnist_results()

    t_values = sort(results[:trajectory_lengths])
    s_values = sort(collect(Set(vcat([collect(keys(d)) for d in values(results[:data])]...))))

    losses = zeros(Float64, length(s_values), length(t_values))
    errors = similar(losses)
    for (i, (s, t)) in enumerate(Base.product(s_values, t_values))
        losses[i] = results[:data][t][s][:loss]
        errors[i] = results[:data][t][s][:loss_error]
    end

    fig = plot_s_graph(s_values, t_values, losses;
        ticks_kwargs=Dict(:round_digits => 0, :power_step => 1)
    )

    # Modify the axis after creation
    ax = content(fig[1, 1])
    ax.xticks = ([5, 50], ["5", "50"])
    ax.xminorgridvisible = true
    ax.yminorgridvisible = true
    ax.xgridvisible = true
    ax.ygridvisible = true

    return fig
end

function plot_mnist_accuracy_graph()
    results = get_mnist_results()

    t_values = sort(results[:trajectory_lengths])
    s_values = sort(collect(Set(vcat([collect(keys(d)) for d in values(results[:data])]...))))

    accuracies = zeros(Float64, length(s_values), length(t_values))
    accuracy_errors = similar(accuracies)
    for (i, (s, t)) in enumerate(Base.product(s_values, t_values))
        accuracies[i] = results[:data][t][s][:accuracy] * 100
        accuracy_errors[i] = results[:data][t][s][:accuracy_error] * 100
    end

    # Note: plot_s_graph uses log scale by default, we need to create custom for linear y-scale
    defaults = get_plot_defaults()
    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i) / (length(t_values)) * 256))] for i in 1:length(t_values)]

    fig = Figure(; size=defaults[:size])
    ax = Axis(fig[1, 1],
        xlabel=L"s",
        ylabel="Accuracy (%)",
        xscale=log10,
        xticks=([5, 50], ["5", "50"]),
        yticks=[10, 50, 100],
        limits=(nothing, nothing, 0, 100),
        xminorgridvisible=true,
        yminorgridvisible=true,
        xgridvisible=true,
        ygridvisible=true
    )

    marker_shapes = [:circle, :rect, :dtriangle, :utriangle, :diamond, :star5]
    for (i, τ) in enumerate(t_values)
        lines!(ax, s_values, accuracies[:, i],
            color=colors[i],
            linewidth=defaults[:linewidth],
            linestyle=:dash,
            label="τ=$τ"
        )
        scatter!(ax, s_values, accuracies[:, i],
            color=colors[i],
            marker=marker_shapes[mod1(i, length(marker_shapes))],
            markersize=defaults[:markersize]
        )
    end

    axislegend(ax, position=:lt, framevisible=false)

    return fig
end


function plot_and_save_mnist_s_graph()
    fig = plot_mnist_s_graph()

    save(get_mnist_results_figure_s_save_path(), fig)
    nothing
end


function plot_and_save_mnist_accuracy_graph()
    fig = plot_mnist_accuracy_graph()

    save(get_mnist_results_figure_accuracy_save_path(), fig)
    nothing
end

function plot_combined_mnist_graph()
    defaults = get_plot_defaults(; columns=2, height_ratio=1 / 3)

    fig = Figure(; size=defaults[:size])

    # Create subfigures for the three main components
    # Images take 15% width, losses 35%, accuracy 35%
    gl_images = fig[1, 1] = GridLayout()
    gl_losses = fig[1, 2] = GridLayout()
    gl_accuracy = fig[1, 3] = GridLayout()

    # Adjust column widths
    colsize!(fig.layout, 1, Relative(0.15))
    colsize!(fig.layout, 2, Relative(0.425))
    colsize!(fig.layout, 3, Relative(0.425))

    # Create image plots
    images = get_examples((0:9)...)
    for (idx, img) in enumerate(images)
        row = (idx - 1) ÷ 2 + 1
        col = (idx - 1) % 2 + 1
        ax = Axis(gl_images[row, col], aspect=DataAspect())
        heatmap!(ax, rotr90(img), colormap=:grays)
        hidedecorations!(ax)
        hidespines!(ax)
    end

    # Add losses plot
    results = get_mnist_results()
    t_values = sort(results[:trajectory_lengths])
    s_values = sort(collect(Set(vcat([collect(keys(d)) for d in values(results[:data])]...))))
    losses = zeros(Float64, length(s_values), length(t_values))
    for (i, (s, t)) in enumerate(Base.product(s_values, t_values))
        losses[i] = results[:data][t][s][:loss]
    end

    s_ticks = calculate_axis_ticks(minimum(s_values), maximum(s_values); round_digits=0, power_step=1)
    loss_ticks = calculate_axis_ticks(minimum(losses), maximum(losses); round_digits=0, power_step=1)

    ax_loss = Axis(gl_losses[1, 1],
        xlabel=L"s",
        ylabel=L"\mathbb{E}[\overline{\mathcal{L}}] / \tau",
        xscale=log10,
        yscale=log10,
        xticks=([5, 50], ["5", "50"]),
        yticks=loss_ticks,
        xgridvisible=true,
        ygridvisible=true,
        xminorgridvisible=true,
        yminorgridvisible=true
    )

    color_palette = ColorSchemes.matter
    colors = [color_palette[Int(round((i) / (length(t_values)) * 256))] for i in 1:length(t_values)]
    for (i, τ) in enumerate(t_values)
        lines!(ax_loss, s_values, losses[:, i], color=colors[i], linewidth=defaults[:linewidth])
        scatter!(ax_loss, s_values, losses[:, i], color=colors[i], markersize=defaults[:markersize])
    end

    # Add accuracy plot
    accuracies = zeros(Float64, length(s_values), length(t_values))
    for (i, (s, t)) in enumerate(Base.product(s_values, t_values))
        accuracies[i] = results[:data][t][s][:accuracy] * 100
    end

    ax_acc = Axis(gl_accuracy[1, 1],
        xlabel=L"s",
        ylabel="Accuracy (%)",
        xscale=log10,
        xticks=([5, 50], ["5", "50"]),
        yticks=[10, 50, 100],
        limits=(nothing, nothing, 0, 100),
        xgridvisible=true,
        ygridvisible=true,
        xminorgridvisible=true,
        yminorgridvisible=true
    )

    for (i, τ) in enumerate(t_values)
        lines!(ax_acc, s_values, accuracies[:, i], color=colors[i], linewidth=defaults[:linewidth], label="τ=$τ")
        scatter!(ax_acc, s_values, accuracies[:, i], color=colors[i], markersize=defaults[:markersize])
    end

    Legend(gl_accuracy[1, 2], ax_acc, framevisible=false)

    # Add labels
    Label(gl_images[0, :], L"(a)", tellwidth=false, halign=:left)
    Label(gl_losses[0, :], L"(b)", tellwidth=false, halign=:left)
    Label(gl_accuracy[0, :], L"(c)", tellwidth=false, halign=:left)

    return fig
end

function plot_and_save_combined_mnist()
    fig = plot_combined_mnist_graph()

    save(get_combined_mnist_graph_save_path(), fig)
    nothing
end
