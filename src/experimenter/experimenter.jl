module Experimenter

include("experiment.jl")
include("database.jl")

export create_db, Experiment, Trial
export LinearVariable, LogLinearVariable, RepeatVariable, IterableVariable

end