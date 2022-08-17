using MLDatasets
using Plots
using Plots.PlotMeasures
using Images
using Flux
using Statistics
using DataFrames
using NNE.MNISTTraining
using NNE.Experimenter
using Base.Iterators
include("plotting_style.jl")

function get_examples(digits...)
    labels = MNIST.trainlabels(1:100)
    indices = [findlast(labels .== d) for d in digits]
    data = MNIST.traintensor(Float32, indices)
    image_data = [Gray.(reshape(data[:, :, i], 28, 28)') for i = 1:length(indices)]
    return image_data
end

function create_image_plots(digits...)
    defaults = get_plot_defaults()

    letters = ["($c)" for c in ('a':'z')[1:length(digits)]]
    images = get_examples(digits...)
    plts = []
    for (img, l) in zip(images, letters)
        plt = plot(img; ticks=false, title=l, titleloc=:left, defaults)
        push!(plts, plt)
    end

    plt = plot(plts...)
    return plt
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
function plot_s_vs_loss(df::DataFrame; max_loss_samples=typemax(Int))
    trajectory_lengths = sort(collect(Set(df.τ)))
    plt = plot(;)
    marker_shapes = (:circle, :rect, :dtriangle, :utriangle, :diamond)
    for (i, t) in enumerate(trajectory_lengths)
        sub_df = df[df.τ .== t, :]
        s_vals = sort(collect(Set(sub_df.s)))
        losses = Float64[]
        errors = Float64[]
        for s in s_vals
            repeat_ls = (x-> length(x) > max_loss_samples ? x[end-max_loss_samples+1:end] : x).(sub_df[sub_df.s .== s, :losses])
            num_repeats = length(repeat_ls)
            push!(losses, mean(mean(ls) for ls in repeat_ls)/t)
            # errors_repeats = [std(ls)/sqrt(length(ls)) for ls in sub_df[sub_df.s .== s, :losses]]
            # total_error = sqrt(sum(x->x*x, errors_repeats))/num_repeats/t
            total_error = std(mean(ls) for ls in repeat_ls)/t/sqrt(num_repeats)
            push!(errors, total_error)
        end
        scatter!(plt, s_vals, losses; label="τ=$t", yerror=errors, markershape=marker_shapes[(i-1)%length(marker_shapes)+1])
    end
    plot!(plt; xscale=:log10, yscale=:log10)
    xlabel!(plt, "s")
    ylabel!(plt, "<L>/τ")
    return plt
end
function prepare_trials_df!(trials_df::DataFrame)
    insertcols!(trials_df, :losses => (x->x[:observations]).(trials_df.results))
    insertcols!(trials_df, :acceptances => (x->x[:observations]).(trials_df.results))
    insertcols!(trials_df, :s => (x->x[:s]).(trials_df.configuration))
    insertcols!(trials_df, :τ => (x->x[:τ]).(trials_df.configuration))
    insertcols!(trials_df, :σ => (x->x[:σ]).(trials_df.configuration))
end

function plot_avg_loss(results, new_plot=true; should_scale_x=false, kwargs...)
    losses = (x->x[:observations]).(results)
    med_duration = median((x->x[:duration].value).(results)) ./ 1000.0
    tau = mean((x->x[:τ]).(results))
    mean_loss = mean(losses) / tau
    std_loss = std(losses) / sqrt(length(losses)) / tau
    plot_fn = new_plot ? plot : plot!
    x_scale = should_scale_x ? LinRange(0, med_duration, length(mean_loss)) : 1:length(mean_loss)
    plt = plot_fn(x_scale, mean_loss; ribbon=(std_loss, std_loss), legend=false, kwargs...)
    xlabel!(should_scale_x ? "Runtime (s)" : "Epochs")
    ylabel!("Mean Loss")
    return plt
end

function plot_avg_loss_compared(trials; max_y_lim=nothing, kwargs...)
    ts = sort(collect(Set(t.configuration[:τ] for t in trials)))
    ss = sort(collect(Set(t.configuration[:s] for t in trials)))
    split_trials = [[tr for tr in trials if tr.configuration[:τ]==t && tr.configuration[:s]==s] for (s, t) in product(ss, ts)]


    plts = []
    for (j, s) in enumerate(ss)
        plt = nothing
        for (i, t) in enumerate(ts)
            plt = plot_avg_loss([trial.results for trial in split_trials[j, i]], (i==1); label="τ=$t", legend=:outerright, kwargs...)
            title!(plt, "($(Char(96+j)))")
            if !isnothing(max_y_lim)
                ylims!(plt, 0, max_y_lim)
            end
        end
        plot!(plt; titlelocation=:left)
        push!(plts, plt)
    end
    return plot(plts...; layout=(length(plts), 1), dpi=300, size=(600, 1200), left_margin = [10mm 0mm])
end

function conv_1d(y, w=500, sigma=100.0)
    f(x) = exp(-0.5 * x * x / (sigma * sigma)) / (sigma * sqrt(2*π))
    kernel = f.(collect(-w:w))
    kernel = kernel ./ sum(kernel)
    conv_y = similar(y)
    for i in eachindex(y)
        min_i = max(1, i-w)
        max_i = min(length(y), i+w)
        kernel_r = (w-(i-min_i)+1):(w+(max_i-i)+1)
        r = (min_i:max_i)
        norm_kernel = kernel[kernel_r]
        norm_kernel ./= sum(norm_kernel)
        conv_y[r] .= sum(y[r].*norm_kernel)
    end
    return conv_y
end

function plot_acceptance(results, new_plot=true; should_scale_x=false, kwargs...)
    acceptances = (x->Float64.(x[:acceptances])).(results)
    conv_acceptances = (x->conv_1d(x)).(acceptances)
    med_duration = median((x->x[:duration].value).(results)) ./ 1000.0
    mean_acceptances = mean(conv_acceptances)
    std_acceptances = std(conv_acceptances)
    plot_fn = new_plot ? plot : plot!
    x_scale = should_scale_x ? LinRange(0, med_duration, length(mean_acceptances)) : 1:length(mean_acceptances)
    plt = plot_fn(x_scale, mean_acceptances; ribbon=(std_acceptances, std_acceptances), legend=false, kwargs...)
    xlabel!(should_scale_x ? "Runtime (s)" : "Epochs")
    ylabel!("Mean Acceptance")
    return plt
end
