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
    configuration=config
);

db = open_db("experiments.db")

push!(db, experiment)

for trial in experiment
    push!(db, trial)

    println("Done trial $(string(trial.id))")
end

trials = get_trials(db, "d3ba4c4c-471d-46b3-906c-564a2f4910f7")

trials_list = collect(experiment)

experiment = get_experiment(db, "d3ba4c4c-471d-46b3-906c-564a2f4910f7")
