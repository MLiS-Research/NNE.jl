module Interfaces

unimplemented() = error("Unimplemented.")

abstract type AbstractModel end
parameters(model::AbstractModel) = unimplemented()

abstract type AbstractClassificationModel <: AbstractModel end
predict(model::AbstractClassificationModel) = unimplemented()

abstract type AbstractDataset end
features(dataset::AbstractDataset) = unimplemented()

abstract type AbstractClassificationDataset <: AbstractDataset end
labels(dataset::AbstractClassificationDataset) = unimplemented()

abstract type AbstractRegressionDataset <: AbstractDataset end
targets(dataset::AbstractRegressionDataset) = unimplemented()


abstract type AbstractClassificationEnsemble <: AbstractClassificationModel end
Base.length(ensemble::AbstractEnsemble) = unimplemented()
"""
Returns an array of models (subtypes of AbstractModel) representing the ensemble.
"""
models(ensemble::AbstractEnsemble) = unimplemented()

end