using Revise
using NNE
using NNE.Experimenter


config = Dict{Symbol,Any}(
    :epochs => IterableVariable([100, 200, 300]),
    :s => LinearVariable(1.0, 10.0, 5),
    :sigma => LogLinearVariable(0.1, 100.0, 12),
    :problem => "mnist"
);

experiment = Experiment(
    include_file="mnist.jl",
    code="run()",
    name="Test Experiment",
    configuration=config
);

db = open_db("experiments.db")

push!(db, experiment)

for trial in experiment
    push!(db, trial)

    println("Done trial $(string(trial.id))")
end

trials = get_trials(db, experiment.id)

