using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

original_experiment = get_experiment_by_name(db, "MNIST Worker Overlap 3")

config = original_experiment.configuration

config[:epochs] = 50_000
config[:restore_from_complete_trial_id] = MatchIterableVariable([trial.id for trial in original_experiment])
config[:save_final_snapshot] = true
config[:snapshot_every_n] = 15_000
config[:max_perturb_models] = 1
config[:snapshot_label] = "Restoration Extended Snapshot"

experiment = Experiment(
    name="MNIST Restore Experiment 3",
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

@execute experiment db DistributedMode
