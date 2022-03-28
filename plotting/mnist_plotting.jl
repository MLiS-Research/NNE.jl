using MLDatasets
using Plots
using Images
using Flux
using Statistics
using NNE.MNISTTraining
include("plotting_style.jl")

function get_examples(digits...)
    labels = MNIST.trainlabels(1:100)
    indices = [findlast(labels.==d) for d in digits]
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
    model = generate_mnist_model(;outputs)
    _, re = Flux.destructure(model)
    if info_dict[:τ] == 1
        model = re(info_dict[:final_state]) # reconstruct model from params
        return [model |> device]
    end

    models = [re(state)|>device for state in info_dict[:final_state]]
    return models
end

function measure_train_accuracy(info_dict; device=cpu, outputs=2)
    models = reconstruct_mnist_models(info_dict; outputs, device)
    features = info_dict[:features] |> device
    labels = info_dict[:labels] |> cpu
    accuracies = [Flux.mean(reshape((x->x[1]-1).(argmax(m(features) |> cpu, dims=1)), :) .== labels) for m in models]
    return accuracies
end