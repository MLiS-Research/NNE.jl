using MLDatasets
using Plots
using Images
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