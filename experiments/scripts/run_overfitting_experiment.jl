using NNE
using Experimenter
using Experimenter: @execute
using Logging

experiment_name = "CIFAR"

config = Dict{Symbol,Any}(
    :s => IterableVariable([50.0, 100.0, 150.0, 200.0]),
    :tau => IterableVariable([16, 32, 64, 96]),
    :samples_per_label => 200,
    :dataset_name => :CIFAR10,
    :sigma => 0.05,
    :param_frac_changed => 0.2,
    :device => :gpu,
    :epochs => 40000000,
    :seed => 46938723,
    :use_progress => true,
    :logging_dirname => "cifar_nne"
)

experiment = Experiment(
    name=experiment_name,
    include_file="overfitting_experiment.jl",
    function_name="run_experiment",
    configuration=deepcopy(config)
)

db = open_db("experiments.db", joinpath(pwd(), "results", "overfitting"))

experiment = restore_from_db(db, experiment)

@execute experiment db DistributedMode true
