module Ensembles
import ..Interfaces
import ..Utils

struct ClassificationEnsemble{M<:Interfaces.AbstractClassificationModel} <: Interfaces.AbstractClassificationEnsemble
    models::Vector{M}
    num_classes::Int
    class_type::DataType
end

function ClassificationEnsemble(base::Interfaces.AbstractClassificationModel, ensemble_parameters)
    @assert length(ensemble_parameters) > 1 "Must have at least one set of parameters"
    return ClassificationEnsemble(map(p -> create_from(base, p), ensemble_parameters), Interfaces.num_classes(base), Interfaces.class_type(base))
end

Base.length(ensemble::ClassificationEnsemble) = length(ensemble.models)
Interfaces.models(ensemble::ClassificationEnsemble) = ensemble.models
Interfaces.class_type(ensemble::ClassificationEnsemble) = ensemble.class_type
Interfaces.num_classes(ensemble::ClassificationEnsemble) = ensemble.num_classes

function Interfaces.predict(ensemble::Interfaces.AbstractClassificationEnsemble, features)
    class_type = Interfaces.class_type(ensemble)
    num_classes = Interfaces.num_classes(ensemble)
    total_votes = similar(features, class_type, (num_classes, size(features, ndims(features))))
    fill!(total_votes, zero(eltype(total_votes)))

    models = Interfaces.models(ensemble)
    for model in models
        predictions = Interfaces.predict(model, features)
        Utils.vote_for_label!(total_votes, predictions)
    end
    ensemble_predictions = Utils.logits_to_predictions(total_votes)
    return ensemble_predictions
end



end