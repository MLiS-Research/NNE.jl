using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute

num_repeats = 15

config = Dict{Symbol,Any}(
    :s => LogLinearVariable(0.01, 5.0, 7),
    :τ => IterableVariable([1, 2, 4, 8, 16]),
    :n_samples => 2048,
    :outputs => 2,
    :σ => 0.05,
    :fraction_to_include => 0.25,
    :device => :gpu,
    :epochs => 300_000,
    :dataset_seed => 46938723,
    :model_seed => 13124,
    :repeat_number => IterableVariable(1:num_repeats),
)

experiment = Experiment(
    name="MNIST Experiment 200k",
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

@execute experiment db DistributedMode
