module Utils
using CUDA

# Predictions

function logits_to_predictions(logits::AbstractArray)
    @assert ndims(logits) == 2
    predictions = Vector{Int}(undef, size(logits, 2))
    @inbounds for i in eachindex(predictions)
        current_max = 1
        max_value = typemin(eltype(logits))
        for j in axes(logits, 1)
            v = logits[j, i]
            if v > max_value
                current_max = j
                max_value = v
            end
        end
        predictions[i] = current_max
    end
    return predictions
end
function logits_to_predictions(logits::CuArray)
    @assert ndims(logits) == 2

    n = size(logits, 2)
    num_threads = min(1024, n)
    num_blocks = cld(n, num_threads)
    predictions = similar(logits, Int64, (n,))
    @cuda blocks = num_blocks threads = num_threads __logits_to_predictions_kernel!(predictions, logits)

    return predictions
end

function __logits_to_predictions_kernel!(predictions, logits)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(logits, 2)
        current_max = 1
        max_value = typemin(eltype(logits))
        for j in 1:size(logits, 1)
            v = logits[j, i]
            if v > max_value
                current_max = j
                max_value = v
            end
        end
        predictions[i] = current_max
    end
    nothing
end

function vote_for_label!(total_votes::AbstractArray, predictions::AbstractArray)
    @inbounds for (i, p) in enumerate(predictions)
        total_votes[p, i] += one(eltype(total_votes))
    end
    nothing
end
function vote_for_label!(total_votes::CuArray, predictions::CuArray)
    @assert ndims(total_votes) == 2
    @assert ndims(predictions) == 1

    n = size(total_votes, 2)
    num_threads = min(1024, n)
    num_blocks = cld(n, num_threads)
    @cuda blocks = num_blocks threads = num_threads __vote_for_label_kernel!(total_votes, predictions)

    return nothing
end
function __vote_for_label_kernel!(total_votes, predictions)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(total_votes, 2)
        p = predictions[i]
        total_votes[p, i] += one(eltype(total_votes))
    end

    return nothing
end

import Flux
import ..Interfaces
import TransitionPathSampling: AbstractObservable, observe, observe!

struct FluxCrossEntropyLossObservable{IM<:Interfaces.AbstractClassificationModel,ID<:Interfaces.AbstractClassificationDataset,OHL<:AbstractArray} <: AbstractObservable
    base_model::IM
    dataset::ID
    onehotlabels::OHL
end

function FluxCrossEntropyLossObservable(base_model::Interfaces.AbstractClassificationModel, dataset::Interfaces.AbstractClassificationDataset)
    num_classes = Interfaces.num_classes(base_model)
    labels = Interfaces.labels(dataset)
    onehotlabels = Flux.onehotbatch(labels, 1:num_classes)

    return FluxCrossEntropyLossObservable(base_model, dataset, onehotlabels)
end

function (obs::FluxCrossEntropyLossObservable)(parameters::AbstractArray)
    new_model = Interfaces.create_from(obs.base_model, parameters)
    logits = Interfaces.logits(new_model, Interfaces.features(obs.dataset))
    return Flux.logitcrossentropy(logits, obs.onehotlabels)
end
function (obs::FluxCrossEntropyLossObservable)(states::AbstractArray{<:AbstractArray})
    return [obs(s) for s in states]
end

function observe(observable::FluxCrossEntropyLossObservable, state::AbstractArray)
    return observable(state)
end
function observe!(cache, observable::FluxCrossEntropyLossObservable, state::AbstractArray, indices)
    for i in indices
        cache[i] = observe(observable, state[i])
    end
    nothing
end

"""
Calculates the accuracy of predictions when compared to their true labels.

Output is ormalised between 0 and 1, with 1 representing 100% accuracy.
"""
function accuracy(true_labels, predicted_labels)
    return sum(true_labels .== predicted_labels) / length(predicted_labels)
end


to_raw(object::DataType) = object
function to_raw(object)
    raw_object = Dict{Symbol,Any}()
    for pname in fieldnames(typeof(object))
        raw_object[pname] = to_raw(getfield(object, pname))
    end
    return raw_object
end
function save_to!(results::Dict{Symbol,Any}, object)
    for pname in fieldnames(typeof(object))
        if haskey(results, pname)
            @info "Saving $(pname) from type $(typeof(object)) to results, but already contains the key $(pname). Overwritting."
        end

        results[pname] = to_raw(getfield(object, pname))
    end
    nothing
end


function to_cpu(array::AbstractArray)
    return deepcopy(array)
end
function to_cpu(array::AbstractArray{T}) where {T<:AbstractArray}
    return [to_cpu(a) for a in array]
end
function to_cpu(array::CuArray)
    return Array(array)
end

end