# Provides support for retrieving the requested dataset, along with some preprocessing
import MLDatasets
import Random
using Logging


struct ImageDataset{T1,T2}
    features::T1
    labels::T2
    img_size::Tuple{Int,Int}
    num_channels::Int
    name::Symbol
end

Base.@kwdef struct PreprocessConfig
    shuffle::Bool = false
    shuffle_rng_seed::Int = 789224195
    max_samples_per_label::Union{Nothing,Int} = nothing
    correct_zero_based::Bool = true
    excluded_samples::Set{Int} = Set{Int}()
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

function load_dataset(name::Symbol; split::DatasetSplit=SplitTrain, config::PreprocessConfig=PreprocessConfig())
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
    end
    labels = labels[indicies_to_keep]

    features = _select_images(features, indicies_to_keep)

    if config.shuffle
        rng = Random.Xoshiro(config.shuffle_rng_seed)
        shuffled_indices = Random.shuffle!(rng, collect(1:length(labels)))
        labels = labels[shuffled_indices]
        features = _select_images(features, shuffled_indices)
    end

    if !isnothing(config.max_samples_per_label)
        class_idxs = map(unique_classes) do class_lbl
            idxs = map((i, _) -> i, filter((_, l) -> l == class_lbl, labels))
            if length(idxs) < config.max_samples_per_label
                @warn "Class index $class_lbl only has $(length(idxs)) images, but aiming for $(config.max_samples_per_label) images."
            else
                idxs = idxs[1:config.max_samples_per_label]
            end
            idxs
        end

        features = reduce(hcat, Iterators.map(idxs -> _select_images_view(features, indicies_to_keep), class_idxs))
        labels = reduce(hcat, Iterators.map(idxs -> @views labels[idxs], class_idxs))

        if config.shuffle # shuffle again
            rng = Random.Xoshiro(config.shuffle_rng_seed)
            shuffled_indices = Random.shuffle!(rng, collect(1:length(labels)))
            labels = labels[shuffled_indices]
            features = _select_images(features, shuffled_indices)
        end
    end

    img_size = (size(features, 1), size(features, 2))
    num_channels = if length(size(features)) == 3
        1
    else
        size(features, 3)
    end


    return ImageDataset(features, labels, img_size, num_channels, name)
end

## API

export DatasetSplit, SplitTest, SplitTrain
export ImageDataset
export PreprocessConfig
export load_dataset