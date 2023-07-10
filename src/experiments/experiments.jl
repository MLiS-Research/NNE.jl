module Experiments

import ..Interfaces
import ..Utils
import TransitionPathSampling as TPS
using Logging
using ProgressBars
include("config.jl")

to_raw(object::DataType) = object
function to_raw(object)
    raw_object = Dict{Symbol,Any}()
    for pname in propertynames(typeof(object))
        raw_object[pname] = to_raw(getproperty(object, pname))
    end
    return raw_object
end
function save_to!(results::Dict{Symbol,Any}, object)
    for pname in propertynames(typeof(object))
        if haskey(results, pname)
            @info "Saving $(pname) from type $(typeof(object)) to results, but already contains the key $(pname). Overwritting."
        end

        results[pname] = to_raw(getproperty(object, pname))
    end
    nothing
end

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
    results[:final_state] = deepcopy(TPS.get_current_state(solution))
    results[:observations] = deepcopy(solution.observations)
end

function run(config::ExperimentConfig, model::Interfaces.AbstractClassificationModel, dataset::Interfaces.AbstractClassificationDataset)
    results = Dict{Symbol,Any}()
    save_to!(results, config)

    problem = setup_problem(config, model, dataset)
    alg = setup_algorithm(config)

    results[:initial_state] = deepcopy(Interfaces.parameters(model))
    iter = 1:config.epochs
    if config.use_progress
        iter = ProgressBar(iter)
    end
    solution = TPS.solve(problem, alg, iter)
    save_solution!(results, solution)

    return results
end

end