using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

original_experiment = get_experiment_by_name(db, "MNIST Experiment 200k")

config = original_experiment.configuration

config[:epochs] = 1000
config[:restore_from_complete_trial_id] = MatchIterableVariable([trial.id for trial in original_experiment])
config[:save_final_snapshot] = true
config[:snapshot_every_n] = 5000
config[:max_perturb_models] = 1
config[:snapshot_label] = "Restoration Snapshot"

experiment = Experiment(
    name="MNIST Restore Experiment 1",
    include_file="../distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

@execute experiment db DistributedMode
