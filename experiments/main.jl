using Revise
using NNE
using NNE.Experimenter

cd("experiments")

config = Dict{Symbol,Any}(
    :epochs => IterableVariable([100, 200]),
    :sigma => LogLinearVariable(0.1, 100.0, 3),
    :seed => IterableVariable([1234, 4321])
);

experiment = Experiment(
    include_file="setup.jl",
    function_name="generate_random_walk",
    name="Experiment 3",
    configuration=config
);

db = open_db("experiments.db")

runner = Runner(execution_mode=DistributedMode, experiment=experiment, database=db)

execute(runner)

get_trials_by_name(db, "Experiment 3")