using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute
using UUIDs

db = open_db("export_sulis_mnist_results.db", joinpath(pwd(), "results", "large"))

original_experiment = get_experiment_by_name(db, "MNIST Final Results 1")

config = original_experiment.configuration
config[:s] = IterableVariable([s for s in config[:s] if s > 1.0])
config[:epochs] = 150_000

experiment = Experiment(
    name="MNIST Final Results 2",
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

matching_trial_ids = UUID[]
original_trials = [t for t in original_experiment]
for new_trial in experiment
    function does_match(t)
        t.configuration[:repeat_number] != new_trial.configuration[:repeat_number] && return false
        t.configuration[:s] != new_trial.configuration[:s] && return false
        t.configuration[:τ] != new_trial.configuration[:τ] && return false
        
        return true
    end
    matching_id = first([trial.id for trial in original_trials if does_match(t)])
    push!(matching_trial_ids, matching_id)
end

config[:restore_from_complete_trial_id] = MatchIterableVariable(matching_trial_ids)
config[:save_final_snapshot] = true

# Reset the experiment with new configuration
experiment = Experiment(
    name="MNIST Final Results 2",
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)


@execute experiment db DistributedMode


function combine_trials(trials_old, trials_new)
    function does_match(t1, t2)
        t1.configuration[:repeat_number] != t2.configuration[:repeat_number] && return false
        t1.configuration[:s] != t2.configuration[:s] && return false
        t1.configuration[:τ] != t2.configuration[:τ] && return false
        
        return true
    end

    trials_dict = Dict{Tuple, Trial}();
    trial_tuple(t::Trial) = (t.configuration[:s], t.configuration[:τ], t.configuration[:repeat_number])
    for t in trials_old
        trials_dict[trial_tuple(t)] = t
    end
    for t in trials_new
        trials_dict[trial_tuple(t)] = t
    end

    return collect(values(trials_dict))
end