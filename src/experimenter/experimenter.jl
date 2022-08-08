module Experimenter

include("experiment.jl")
include("database.jl")
include("runner.jl")

export open_db, Experiment, Trial
export LinearVariable, LogLinearVariable, RepeatVariable, IterableVariable
export get_experiment, get_experiments, get_trial, get_trials, get_experiment_by_name, complete_trial!, complete_trial_in_global_database, get_trials_by_name
export execute_trial, execute_trial_and_save_to_db_async
export execute, Runner
export SerialMode, MultithreadedMode, DistributedMode
export restore_from_db
export merge_databases!
end