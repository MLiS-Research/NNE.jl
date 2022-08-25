using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute
using Logging

should_extend = true
num_repeats = 3
experiment_name = "MNIST Annealing Ex 9"

config = Dict{Symbol,Any}(
    :s => LogLinearVariable(5.0, 50.0, 5),
    :τ => IterableVariable([1, 2, 4, 8, 16, 32]),
    :n_samples => 2048,
    :outputs => 10,
    :σ => 0.05,
    :fraction_to_include => 0.25,
    :device => :gpu,
    :epochs => 2_500_000,
    :dataset_seed => 46938723,
    :model_seed => 13124,
    :repeat_number => IterableVariable(1:num_repeats),
    :max_perturb_models => 1,
    :save_final_snapshot => true,
    :use_previous_snapshot => true,
    :snapshot_every_n => 100_000,
    :snapshot_label => "Annealed MNIST",
    :start_s => 1.0,
    :annealing_epochs => 100_000
)

experiment = Experiment(
    name=experiment_name,
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=deepcopy(config)
)

function adjust_sigma(trial, σ; cutoff=0.01)
    cross_over = 1.0/(σ^2)
    s_val = trial.configuration[:s]
    t = trial.configuration[:τ]
    if t > 1 && s_val < cutoff * cross_over
        return σ / ((s_val)/(cross_over*cutoff))
    elseif t == 1 && s_val < cross_over
        return σ / ((s_val)/(cross_over))
    else
        return σ
    end
end

config[:σ] = MatchIterableVariable([adjust_sigma(trial, trial.configuration[:σ]) for trial in experiment])

# Recreate experiment with the matched variable
experiment = Experiment(
    name=experiment.name,
    include_file=experiment.include_file,
    function_name=experiment.function_name,
    configuration=deepcopy(config)
)

db = open_db("experiments_new.db", joinpath(pwd(), "results", "large"))

experiment = restore_from_db(db, experiment)
if should_extend
    @info "Extending trials"
    for trial in experiment
        mark_trial_as_incomplete!(db, trial.id);
    end
end

@execute experiment db DistributedMode
