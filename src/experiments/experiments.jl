module Experiments

import ..Interfaces
import ..Utils
import TransitionPathSampling as TPS
using Logging
using ProgressBars
import CUDA: CuArray
import ..Callbacks as CB
include("config.jl")


function construct_initial_state(trajectory_length, model::Interfaces.AbstractModel)
    params = Interfaces.parameters(model)
    if trajectory_length == 1
        return deepcopy(params)
    else
        return map(x -> deepcopy(params), 1:trajectory_length)
    end
end

function setup_algorithm(config::ExperimentConfig)
    if config.trajectory_length == 1
        algorithm = TPS.MetropolisHastings.gaussian_sa_algorithm(
            config.algorithm_config.bias,
            config.algorithm_config.sigma;
            params_changed_frac=config.algorithm_config.parameter_perturb_fraction
        )
        return algorithm
    else
        algorithm = TPS.MetropolisHastings.gaussian_trajectory_algorithm(
            config.algorithm_config.bias,
            config.algorithm_config.sigma;
            params_changed_frac=config.algorithm_config.parameter_perturb_fraction,
            max_width=1,
            chance_to_shoot=(2 / config.trajectory_length)
        )
        return algorithm
    end
end

function setup_problem(config::ExperimentConfig, model::Interfaces.AbstractClassificationModel, dataset::Interfaces.AbstractClassificationDataset)
    observable = Utils.FluxCrossEntropyLossObservable(model, dataset)
    state = construct_initial_state(config.trajectory_length, model)

    if config.trajectory_length == 1
        return TPS.SimulatedAnnealing.SAProblem(observable, state)
    else
        return TPS.DiscreteTrajectory.DTProblem(observable, state)
    end
end

function save_solution!(results, solution::TPS.SimpleSolution)
    results[:final_state] = Utils.to_cpu(TPS.get_current_state(solution))
    results[:observations] = deepcopy(solution.observations)
end

function construct_callbacks(config::ExperimentConfig, model::Interfaces.AbstractClassificationModel, train_dataset::Interfaces.AbstractClassificationDataset, validation_dataset::Interfaces.AbstractClassificationDataset)
    # Extend with saving callbacks
    if isnothing(config.tensorboard_logging_config)
        cb = construct_tb_callback(config, validation_dataset)
        return cb
    end

    return nothing
end
function _train_metrics(config::ExperimentConfig)
    modifier = Float32(1.0 / config.trajectory_length)
    tb_config::TensorboardLoggingConfig = config.tensorboard_logging_config
    loss_metric = tb_config.train_loss_frequency == 0 ? nothing : CB.TrainingLossMetricGatherer(modifier, tb_config.train_loss_frequency)
    accuracy_metric = tb_config.train_accuracy_frequency == 0 ? nothing : CB.TrainingAccuracyMetricGatherer(tb_config.train_accuracy_frequency)
    return loss_metric, accuracy_metric
end
function _validation_metrics(config::TensorboardLoggingConfig, dataset)
    loss_metric = config.validation_loss_frequency == 0 ? nothing : CB.ValidationLossMetricGatherer(dataset, config.validation_loss_frequency)
    accuracy_metric = config.validation_accuracy_frequency == 0 ? nothing : CB.ValidationAccuracyMetricGatherer(dataset, config.validation_accuracy_frequency)
    return loss_metric, accuracy_metric
end
function construct_tb_metrics(config::TensorboardLoggingConfig, ::Nothing)
    loss_metric, accuracy_metric = _train_metrics(config::ExperimentConfig)
    metrics = filter(!isnothing, (loss_metric, accuracy_metric))
    return metrics
end
function construct_tb_metrics(config::TensorboardLoggingConfig, validation_dataset::Interfaces.AbstractClassificationDataset)
    train_metrics = construct_tb_callback(config, nothing)
    validation_metrics = filter(!isnothing, _validation_metrics(config.tensorboard_logging_config, validation_dataset))
    return (train_metrics..., validation_metrics...)
end
function construct_tb_callback(config::ExperimentConfig, validation_dataset::Interfaces.AbstractClassificationDataset)
    metrics = construct_tb_metrics(config, validation_dataset)
    if length(metrics) == 0
        return nothing
    end

    tb_config::TensorboardLoggingConfig = config.tensorboard_logging_config
    cb = CB.TBLoggerCallback(tb_config.path, metrics)
    return cb
end


function run(config::ExperimentConfig,
    model::Interfaces.AbstractClassificationModel,
    dataset::Interfaces.AbstractClassificationDataset;
    validation_dataset::Union{Nothing,Interfaces.AbstractClassificationDataset}=nothing
)
    results = Dict{Symbol,Any}()
    Utils.save_to!(results, config)

    problem = setup_problem(config, model, dataset)
    alg = setup_algorithm(config)

    results[:initial_state] = Utils.to_cpu(TPS.get_initial_state(problem))
    iter = 1:config.epochs
    if config.use_progress
        iter = ProgressBar(iter)
    end

    cb = construct_callbacks(config, model, dataset, validation_dataset)

    solution = TPS.solve(problem, alg, iter; cb=cb)
    save_solution!(results, solution)

    return results
end

export run, ExperimentConfig, AlgorithmConfig

end