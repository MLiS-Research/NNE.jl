using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute

num_repeats = 3
experiment_name = "Perceptron Ex 0"

config = Dict{Symbol,Any}(
    :s => LogLinearVariable(0.01, 1000.0, 9),
    :τ => IterableVariable([1, 2, 4, 8, 16]),
    :σ => IterableVariable([0.2, 1.0, 5.0]),
    :annealing_epochs => 500,
    :epochs => 1000,
    :repeat_number => IterableVariable(1:num_repeats),
    :max_width => 3,
    :initial_s => 1.0
)

experiment = Experiment(
    name=experiment_name,
    include_file="distributed/perceptron_runner.jl",
    function_name="linear_tps",
    configuration=config
)

db = open_db("temp.db", joinpath(pwd(), "results", "large"))

@execute experiment db DistributedMode
