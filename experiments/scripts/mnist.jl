using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute
using Logging

should_extend = false
num_repeats = 3
experiment_name = "MNIST (Test)"

config = Dict{Symbol,Any}(
    :s => LogLinearVariable(5.0, 50.0, 2),
    :τ => IterableVariable([1, 2, 4]),
    :n_samples => 2048,
    :outputs => 10,
    :σ => 0.05,
    :fraction_to_include => 0.25,
    :device => :gpu,
    :epochs => 10_000,
    :dataset_seed => 46938723,
    :model_seed => 13124,
    :repeat_number => IterableVariable(1:num_repeats),
    :max_perturb_models => 1,
    :save_final_snapshot => true,
    :use_previous_snapshot => true,
    :snapshot_every_n => 5_000,
    :snapshot_label => "Annealed MNIST Test",
    :start_s => 1.0,
    :annealing_epochs => 5_000
)

experiment = Experiment(
    name=experiment_name,
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=deepcopy(config)
)

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

experiment = restore_from_db(db, experiment)
if should_extend
    @info "Extending trials"
    for trial in experiment
        mark_trial_as_incomplete!(db, trial.id);
    end
end

@execute experiment db DistributedMode
