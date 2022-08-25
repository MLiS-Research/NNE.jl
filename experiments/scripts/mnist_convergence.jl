using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute

num_repeats = 60

config = Dict{Symbol,Any}(
    :s => IterableVariable([0.25, 0.5, 1.0]),
    :τ => IterableVariable([2, 4, 8]),
    :n_samples => 1024,
    :outputs => 2,
    :σ => 0.05,
    :fraction_to_include => 0.25,
    :device => :gpu,
    :epochs => 100_000,
    :dataset_seed => 46938723,
    :model_seed => 13124,
    :repeat_number => IterableVariable(1:num_repeats),
    :max_perturb_models => 1,
    :save_final_snapshot => true,
    :use_previous_snapshot => true,
    :snapshot_every_n => 50_000,
    :snapshot_label => "Convergence Test"
)

experiment = Experiment(
    name="MNIST Convergence Experiment",
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

# Reset all of the trials
for trial in get_trials_by_name(db, experiment.name)
    mark_trial_as_incomplete!(db, trial.id);
end

@execute experiment db DistributedMode
