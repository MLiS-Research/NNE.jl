function generate_cross_entropy_loss_fn(dataset::ImageDataset, model::ImageModel)
    onehotencoded_labels = Flux.onehotbatch(dataset.labels, 1:model.num_outputs)
    function loss_fn(parameters::AbstractArray)
        flux_model = model.reconstruct_fn(parameters)
        logits = flux_model(dataset.features)
        return Flux.logitcrossentropy(logits, onehotencoded_labels)
    end
    function loss_fn(parameters::AbstractArray{T}) where {T<:AbstractArray}
        return loss_fn.(parameters)
    end
    return loss_fn
end