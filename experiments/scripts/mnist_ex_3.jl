using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute

config = Dict{Symbol,Any}(
    :s => 10.0,
    :τ => 4,
    :n_samples => 256,
    :outputs => 2,
    :σ => 0.05,
    :fraction_to_include => 0.25,
    :device => :gpu,
    :epochs => 10_000,
    :dataset_seed => 46938723,
    :model_seed => 13124,
    :max_perturb_models => 1,
    :save_final_snapshot => true,
    :use_previous_snapshot => true,
    :show_progress => true,
    :snapshot_every_n => 1000,
    :snapshot_label => "Callbacks Test Snapshot"
)

experiment = Experiment(
    name="MNIST Callbacks Test 2",
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

@execute experiment db SerialMode

original_trial = first(get_trials_by_name(db, experiment.name));

mark_trial_as_incomplete!(db, original_trial.id);

@execute experiment db SerialMode

new_trial = first(get_trials_by_name(db, experiment.name));

include("../plotting/mnist_plotting.jl")

plot_avg_loss([new_trial.results])
plot_avg_loss([original_trial.results], false)
plot_acceptance([new_trial.results], true)
plot_acceptance([original_trial.results], false)