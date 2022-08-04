module Experimenter

include("experiment.jl")
include("database.jl")

export open_db, Experiment, Trial
export LinearVariable, LogLinearVariable, RepeatVariable, IterableVariable
export get_experiment, get_experiments, get_trial, get_trials
end