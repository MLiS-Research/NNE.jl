using Revise
using NNE.Experimenter

config = Dict{Symbol,Any}(
    :s => LogLinearVariable(0.01, 5.0, 9),
    :τ => IterableVariable([1, 2, 4, 8, 16]),
    :n_samples => 2048,
    :outputs => 2,
    :σ => 0.05,
    :fraction_to_include => 0.25,
    :device => :gpu,
    :epochs => 100
)

experiment = Experiment(
    name="MNIST Experiment 1",
    include_file="distributed/mnist_runner.jl",
    function_name="map_params_to_trajectory",
    configuration=config
)

db = open_db("experiments.db", joinpath(pwd(), "experiments", "results", "large"))

runner = Runner(execution_mode=SerialMode, experiment=experiment, database=db)

execute(runner)