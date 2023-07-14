using ..Utils
import ..Ensembles
import ..Interfaces
import TransitionPathSampling.Callbacks as CB
import TransitionPathSampling as TPS
import TransitionPathSampling.MetropolisHastings as MH


_get_base_model(obs::Utils.FluxCrossEntropyLossObservable) = obs.base_model
_get_dataset(obs::Utils.FluxCrossEntropyLossObservable) = obs.dataset

_get_model(state::AbstractArray, base_model) = create_from(base_model, state)
_get_model(states::AbstractArray{<:AbstractArray}, base_model) = Ensembles.ClassificationEnsemble(base_model, states)

function _accuracy(state::AbstractArray{<:AbstractArray}, base_model::Interfaces.AbstractClassificationModel, dataset::Interfaces.AbstractClassificationDataset)
    model = _get_model(state, base_model)
    test_predictions = Interfaces.predict(model, Interfaces.features(dataset))
    test_labels = Interfaces.labels(dataset)

    accuracy = Utils.accuracy(test_labels, test_predictions)
    return accuracy
end

Base.@kwdef struct TrainingLossMetricGatherer <: Tensorboard.AbstractMetricGatherer
    modifier::Float32 = 1.0f0
    frequency::Int = 1
end
Tensorboard.tag(::TrainingLossMetricGatherer) = "train/loss"
Tensorboard.frequency(m::TrainingLossMetricGatherer) = m.frequency
function Tensorboard.gather(gatherer::TrainingLossMetricGatherer, deps::CB.SolveDependencies)
    return _get_loss(deps.cache) * gatherer.modifier
end
_get_loss(cache::MH.AbstractMetropolisHastingsCache) = MH.get_last_observation(cache)
_get_loss(cache) = error("Cache does not have an implemented method for retrieving the last loss.")

Base.@kwdef struct TrainingAccuracyMetricGatherer <: Tensorboard.AbstractMetricGatherer
    frequency::Int = 1
end
Tensorboard.tag(::TrainingAccuracyMetricGatherer) = "train/accuracy"
Tensorboard.frequency(m::TrainingAccuracyMetricGatherer) = m.frequency
function Tensorboard.gather(::TrainingAccuracyMetricGatherer, deps::CB.SolveDependencies)
    state = TPS.get_current_state(deps.solution)
    obs = TPS.get_observable(deps.problem)
    base_model = _get_base_model(obs)
    dataset = _get_dataset(obs)
    return _accuracy(state, base_model, dataset)
end

Base.@kwdef struct ValidationAccuracyMetricGatherer{D<:Interfaces.AbstractClassificationDataset} <: Tensorboard.AbstractMetricGatherer
    dataset::D
    frequency::Int = 1
end
Tensorboard.tag(::ValidationAccuracyMetricGatherer) = "validation/accuracy"
Tensorboard.frequency(m::ValidationAccuracyMetricGatherer) = m.frequency
function Tensorboard.gather(gatherer::ValidationAccuracyMetricGatherer, deps::CB.SolveDependencies)
    state = TPS.get_current_state(deps.solution)
    obs = TPS.get_observable(deps.problem)
    base_model = _get_base_model(obs)
    dataset = gatherer.dataset
    return _accuracy(state, base_model, dataset)
end

Base.@kwdef struct ValidationLossMetricGatherer{D<:Interfaces.AbstractClassificationDataset} <: Tensorboard.AbstractMetricGatherer
    dataset::D
    frequency::Int = 1
end
Tensorboard.tag(::ValidationLossMetricGatherer) = "validation/loss"
Tensorboard.frequency(m::ValidationLossMetricGatherer) = m.frequency
function Tensorboard.gather(gatherer::ValidationLossMetricGatherer, deps::CB.SolveDependencies)
    state = TPS.get_current_state(deps.solution)
    obs = TPS.get_observable(deps.problem)
    base_model = _get_base_model(obs)
    validation_obs = Utils.FluxCrossEntropyLossObservable(base_model, gatherer.dataset)
    losses = validation_obs(state)
    return sum(losses) / length(losses) # Return mean over ensemble
end