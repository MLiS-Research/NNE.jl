# Provides support for retrieving the requested dataset, along with some preprocessing
import MLDatasets
import Random
using Logging
using Flux

struct ImageDataset{T1,T2}<:Interfaces.AbstractClassificationDataset
    features::T1
    labels::T2
    img_size::Tuple{Int,Int}
    num_channels::Int
    name::Symbol
end
Interfaces.features(dataset::ImageDataset) = dataset.features
Interfaces.labels(dataset::ImageDataset) = dataset.labels

Base.@kwdef struct PreprocessConfig{T<:Union{AbstractArray,Set}}
    shuffle::Bool = false
    shuffle_rng_seed::Int = 789224195
    max_samples_per_label::Union{Nothing,Int} = nothing
    correct_zero_based::Bool = true
    excluded_samples::T = Set{Int}()
end

function _get_dataset_generator(name::Symbol)
    if name == :MNIST
        return MLDatasets.MNIST
    elseif name == :CIFAR10
        return MLDatasets.CIFAR10
    else
        error("Unrecognised dataset $name")
    end
end

@enum DatasetSplit SplitTest SplitTrain

function _split_to_symbol(s::DatasetSplit)
    if s == SplitTest
        return :test
    else
        return :train
    end
end
_select_images(f, indices) = f[((i -> 1:i).(size(f)[begin:end-1]))..., indices]
_select_images_view(f, indices) = @views f[((i -> 1:i).(size(f)[begin:end-1]))..., indices]

function load_dataset(name::Symbol; split::DatasetSplit=SplitTrain, config::PreprocessConfig=PreprocessConfig(), device=Flux.cpu)
    dataset_generator = _get_dataset_generator(name)
    dataset = dataset_generator(; split=_split_to_symbol(split))

    labels = dataset.targets # Assumes these are 0-based
    features = dataset.features

    # Preprocessing
    if config.correct_zero_based
        labels .+= one(eltype(labels))
    end
    @assert eltype(labels) <: Integer
    unique_classes = sort(unique(labels))
    max_class = maximum(labels)
    @assert all((1:max_class) .== unique_classes) # Assume no missing classes

    if length(config.excluded_samples) > 0
        indicies_to_keep = map(l -> !(l in config.excluded_samples), labels)
        unique_classes = filter(x -> !(x in config.excluded_samples), unique_classes)
        labels = labels[indicies_to_keep]
        features = _select_images(features, indicies_to_keep)
    end


    if config.shuffle
        rng = Random.Xoshiro(config.shuffle_rng_seed)
        shuffled_indices = Random.shuffle!(rng, collect(1:length(labels)))
        labels = labels[shuffled_indices]
        features = _select_images(features, shuffled_indices)
    end

    if !isnothing(config.max_samples_per_label)
        class_idxs = map(unique_classes) do class_lbl
            idxs = [i for (i, lbl) in enumerate(labels) if lbl == class_lbl]
            if length(idxs) < config.max_samples_per_label
                @warn "Class index $class_lbl only has $(length(idxs)) images, but aiming for $(config.max_samples_per_label) images."
            else
                idxs = idxs[1:config.max_samples_per_label]
            end
            idxs
        end
        image_views = map(idxs -> _select_images_view(features, idxs), class_idxs)
        total_images = sum(x->size(x, ndims(x)), image_views)
        new_features = similar(features, eltype(features), ((size(features)[begin:end-1])..., total_images))
        image_offset = 1
        for img_view in image_views
            n_images = size(img_view, ndims(img_view))
            new_features[(1:d for d in size(features)[begin:end-1])..., image_offset:(image_offset+n_images-1)] .= img_view
            image_offset += n_images
        end
        features = new_features
        labels = reduce(vcat, map(idxs -> view(labels, idxs), class_idxs))

        if config.shuffle # shuffle again
            rng = Random.Xoshiro(config.shuffle_rng_seed)
            shuffled_indices = Random.shuffle!(rng, collect(1:length(labels)))
            labels = labels[shuffled_indices]
            features = _select_images(features, shuffled_indices)
        end
    end

    img_size = (size(features, 1), size(features, 2))
    num_channels = if length(size(features)) == 3
        features = reshape(features, img_size..., 1, :) # Make sure to convert the shape to be consistent
        1
    else
        size(features, 3)
    end


    return ImageDataset(features |> device, labels |> device, img_size, num_channels, name)
end

## API

export DatasetSplit, SplitTest, SplitTrain
export ImageDataset
export PreprocessConfig
export load_dataset