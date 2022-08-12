using MLDatasets
using Plots
using Images
using Flux
using Statistics
using NNE.MNISTTraining
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

function plot_avg_loss(results, new_plot=true; should_scale_x=false, kwargs...)
    losses = (x->x[:observations]).(results)
    med_duration = median((x->x[:duration].value).(results)) ./ 1000.0
    mean_loss = mean(losses)
    std_loss = std(losses)
    plot_fn = new_plot ? plot : plot!
    x_scale = should_scale_x ? LinRange(0, med_duration, length(mean_loss)) : 1:length(mean_loss)
    plt = plot_fn(x_scale, mean_loss; ribbon=(std_loss, std_loss), legend=false, kwargs...)
    xlabel!(should_scale_x ? "Runtime (s)" : "Epochs")
    ylabel!("Mean Loss")
    return plt
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
    ylabel!("Mean Loss")
    return plt
end
