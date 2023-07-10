struct ImagePredictionsFn{IM <: ImageModel, ID <: ImageDataset}
    base_model::IM
    dataset::ID
end

function (preds_fn::ImagePredictionsFn)(ensemble_parameters::AbstractArray{T}) where {T<:AbstractArray}
    model = preds_fn.base_model
    dataset = preds_fn.dataset
    total_votes = similar(dataset.labels, Int64, (model.num_outputs, length(dataset.labels)))
    fill!(total_votes, zero(eltype(total_votes)))
    for parameters in ensemble_parameters
        flux_model = model.reconstruct_fn(parameters)
        logits = flux_model(dataset.features)
        predictions = logits_to_predictions(logits)
        vote_for_label!(total_votes, predictions)
    end
    ensemble_predictions = logits_to_predictions(total_votes) # Votes act like logits, where we just do an argmax
    return ensemble_predictions
end
function (preds_fn::ImagePredictionsFn)(parameters::AbstractArray)
    model = preds_fn.base_model
    dataset = preds_fn.dataset
    flux_model = model.reconstruct_fn(parameters)
    logits = flux_model(dataset.features)
    predictions = logits_to_predictions(logits)
    return predictions
end

using CUDA
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
    @cuda blocks=num_blocks threads=num_threads __vote_for_label_kernel!(total_votes, predictions)

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
    @cuda blocks=num_blocks threads=num_threads __logits_to_predictions_kernel!(predictions, logits)

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