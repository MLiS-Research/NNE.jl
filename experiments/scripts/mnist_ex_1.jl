using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute

num_repeats = 1

config = Dict{Symbol,Any}(
    :s => LogLinearVariable(5.0, 50.0, 5),
    :τ => IterableVariable([1, 2, 4, 8, 16]),
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
    :snapshot_every_n => 250_000,
    :snapshot_label => "Full MNIST Long"
)

experiment = Experiment(
    name="MNIST Full 1",
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

db = open_db("experiments_new.db", joinpath(pwd(), "results", "large"))

@execute experiment db DistributedMode
