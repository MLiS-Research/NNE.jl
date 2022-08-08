using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute

num_repeats = 4

config = Dict{Symbol,Any}(
    :s => 1.0,
    :τ => 8,
    :n_samples => 1024,
    :outputs => 2,
    :σ => 0.05,
    :fraction_to_include => 0.25,
    :device => :gpu,
    :epochs => 50_000,
    :dataset_seed => 46938723,
    :model_seed => 13124,
    :repeat_number => IterableVariable(1:num_repeats),
    :max_perturb_models => IterableVariable([nothing, 1]),
)

experiment = Experiment(
    name="MNIST Efficiency Test 2",
    include_file="../distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

@execute experiment db DistributedMode
