using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute
using Logging

epochs = 1_500_000
existing_experiment_name = "MNIST Annealing Ex 11"
experiment_name = "MNIST Annealing Ex 12"

db = open_db("experiments_new.db", joinpath(pwd(), "results", "large"))

existing_experiment = get_experiment_by_name(db, existing_experiment_name)

function generate_config(experiment::Experiment, num_epochs::Int)
    config = deepcopy(experiment.configuration)

    delete!(config, :start_s)
    delete!(config, :annealing_epochs)
    config[:restore_from_complete_trial_id] = MatchIterableVariable([trial.id for trial in experiment])
    config[:σ] = 0.05
    config[:label] = "Annealed MNIST Extended"
    config[:epochs] = num_epochs
    return config
end

config = generate_config(existing_experiment, epochs)

experiment = Experiment(
    name=experiment_name,
    include_file=existing_experiment.include_file,
    function_name=existing_experiment.function_name,
    configuration=config
)

@execute experiment db DistributedMode
