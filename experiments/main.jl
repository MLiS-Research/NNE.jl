using Revise
using NNE
using NNE.Experimenter


config = Dict{Symbol,Any}(
    :epochs => IterableVariable([100, 200]),
    :sigma => LogLinearVariable(0.1, 100.0, 3),
    :seed => IterableVariable([1234, 4321])
);

experiment = Experiment(
    include_file="setup.jl",
    function_name="generate_random_walk",
    name="Experiment 1",
    configuration=config
);

db = open_db("experiments.db")

runner = Runner(execution_mode=SerialMode, experiment=experiment, database=db)

execute(runner)

results = (x -> x.results).(get_trials(db, experiment.id))