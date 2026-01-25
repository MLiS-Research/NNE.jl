module Interfaces

unimplemented() = error("Unimplemented.")

abstract type AbstractModel end
parameters(model::AbstractModel) = unimplemented()
create_from(basemodel::AbstractModel, new_parameters) = unimplemented()
predict(model::AbstractModel, features) = unimplemented()

abstract type AbstractClassificationModel <: AbstractModel end
logits(::AbstractClassificationModel, features) = unimplemented()
class_type(::AbstractClassificationModel) = unimplemented()
num_classes(::AbstractClassificationModel) = unimplemented()

abstract type AbstractDataset end
features(dataset::AbstractDataset) = unimplemented()

abstract type AbstractClassificationDataset <: AbstractDataset end
labels(dataset::AbstractClassificationDataset) = unimplemented()

abstract type AbstractRegressionDataset <: AbstractDataset end
targets(dataset::AbstractRegressionDataset) = unimplemented()


abstract type AbstractClassificationEnsemble <: AbstractClassificationModel end
Base.length(ensemble::AbstractClassificationEnsemble) = unimplemented()
"""
Returns an array of models (subtypes of AbstractModel) representing the ensemble.
"""
models(ensemble::AbstractClassificationEnsemble) = unimplemented()
logits(::AbstractClassificationEnsemble, features) = error("`logits` function is unsupported by ensembles.")

end