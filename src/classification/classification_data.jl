using Flux

struct TPSClassificationDataset{S,T,R}
    features::S
    labels::T
    labels_one_hot::R
end

"""
    construct_cross_entropy_loss_fn(dataset)

Generates a loss function which maps predicted logits to a cross-entropy loss function defined by the dataset labels.
"""
function construct_cross_entropy_loss_fn(dataset::TPSClassificationDataset)
    # Wrap the loss function with the output of the logits
    function loss_fn(logits)
        return Flux.Losses.logitcrossentropy(logits, dataset.labels_one_hot)
    end
    return loss_fn
end

"""
    construct_cross_entropy_loss_fn(model, dataset)

Generates a loss function which maps a set of model parameters to a cross entropy loss defined by the dataset.
"""
function construct_cross_entropy_loss_fn(model, dataset)
    _, re = Flux.destructure(model)
    logit_loss_fn = construct_cross_entropy_loss_fn(dataset)
    function loss_fn(state)
        mdl = re(state)
        return logit_loss_fn(mdl(dataset.features))
    end
    return loss_fn
end
