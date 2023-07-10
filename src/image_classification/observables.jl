import TransitionPathSampling: AbstractObservable, observe, observe!

struct ImageCrossEntropyLossObservable{IM <: ImageModel, ID <: ImageDataset, OHL<:AbstractArray} <: AbstractObservable
    model::IM
    dataset::ID
    onehotlabels::OHL
end

function ImageCrossEntropyLossObservable(dataset::ImageDataset, model::ImageModel)
    onehotlabels = Flux.onehotbatch(dataset.labels, 1:model.num_outputs)

    return ImageCrossEntropyLossObservable(model, dataset, onehotlabels)
end

function (obs::ImageCrossEntropyLossObservable)(parameters::AbstractArray)
    flux_model = obs.model.reconstruct_fn(parameters)
    logits = flux_model(obs.dataset.features)
    return Flux.logitcrossentropy(logits, obs.onehotlabels)
end

function observe(observable::ImageCrossEntropyLossObservable, state::AbstractArray)
    return observable(state)
end
function observe!(cache, observable::SimpleObservable, state::AbstractArray, indices)
    for i in indices
        cache[i] = observe(observable, state[i])
    end
    nothing
end