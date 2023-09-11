using DataFrames
using BSON
using CSV
using CairoMakie
using Flux
using MLDatasets
using NNE
using NNE.ImageClassification
using NNE.Experiments
using Random
using Dates
using ProgressBars
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

function plot_cifar_images(images, labels)
    fig = create_pub_fig(; num_panels=17.8 / 8.6, figure_padding=10)

    for (i, (digit, label)) in enumerate(zip(images, labels))
        row = (i - 1) ÷ 5 + 1
        col = (i - 1) % 5 + 1
        ax = Axis(fig[row, col], aspect=1, title=label)
        Makie.hidedecorations!(ax)
        Makie.image!(ax, reverse(digit; dims=2))
    end

    return fig
end

nne_df = DataFrame(CSV.File(joinpath(@__DIR__, "..", "results", "best_cifar_nne_training_accuracies.csv")))
nne_df[!, :tps_time] = nne_df[!, :Step] ./ (96 * 3666);

adam_accuracy = 49.59
sgd_accuracy = 49.51 # change


dataset_name = :CIFAR10
pre_process_config = PreprocessConfig(shuffle=false, max_samples_per_label=1)
example_dataset = load_dataset(dataset_name; split=SplitTrain, device=Flux.cpu, config=pre_process_config);

function rgb_img(img_values)
    map(Iterators.product(1:size(img_values, 1), 1:size(img_values, 2))) do (i, j)
        Makie.RGBA(img_values[i, j, 1:3]..., 1.0)
    end
end

images = [rgb_img(example_dataset.features[:, :, :, i]) for i in 1:10];
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
images_fig = plot_cifar_images(images, labels)
Makie.save("figures/talk/cifar_images.pdf", images_fig; pt_per_unit=1)

accuracy_fig = begin
    fig = create_pub_fig()
    colour_map = ColorSchemes.matter
    ax = Axis(fig[1, 1]; xlabel="TPS Time", ylabel="Accuracy (%)", ygridvisible=false, xgridvisible=false)
    Makie.hlines!(ax, [adam_accuracy], label="ADAM", linestyle=:dot, linewidth=3, color=colour_map[128])
    Makie.hlines!(ax, [sgd_accuracy], label="SGD", linestyle=:dash, linewidth=3, color=colour_map[64])
    Makie.lines!(ax, nne_df[!, :tps_time], nne_df[!, :Value] .* 100, label="NNE", color=colour_map[256])
    axislegend(ax, position=:rb, orientation=:vertical, framevisible=false, bgcolor=nothing)
    Makie.xlims!(0, maximum(nne_df[!, :tps_time]))
    Makie.ylims!(5, 55)
    fig
end
Makie.save("./figures/talk/cifar_accuracy_comparison.pdf", accuracy_fig; pt_per_unit=1)