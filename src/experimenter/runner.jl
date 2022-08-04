using Base
using Distributed
using ParallelDataTransfer
using Base.Iterators
using Logging

@enum EXECUTEMODE SerialMode MultithreadedMode DistributedMode

Base.@kwdef struct Runner
    execution_mode::EXECUTEMODE
    experiment::Experiment
    database::ExperimentDatabase
end

function execute(runner::Runner)
    # Push to the database
    push!(runner.database, runner.experiment)

    existing_trials = get_trials(runner.database, runner.experiment.id)
    completed_trials = [trial for trial in existing_trials if trial.has_finished]
    completed_uuids = Set(trial.id for trial in completed_trials)
    # Only take unrun trials
    incomplete_trials = [trial for trial in runner.experiment if !(trial.id in completed_uuids)]

    # Push all incomplete trials to the database
    for trial in incomplete_trials
        push!(runner.database, trial)
    end

    prepare_environment(runner)

    run_trials(runner, incomplete_trials)

    nothing
end

function prepare_environment(runner::Runner)
    if runner.execution_mode == DistributedMode
        eval(Meta.parse("@everywhere using Pkg"))
        eval(Meta.parse("@everywhere Pkg.activate(\".\")"))
        eval(Meta.parse("@everywhere using ParallelDataTransfer"))
        eval(Meta.parse("@everywhere using NNE.Experimenter"))
    end

    global include_file = runner.experiment.include_file

    if !ismissing(include_file)
        if runner.execution_mode == DistributedMode
            sendto(workers(), include_file=include_file)
            eval(Meta.parse("@everywhere include(include_file)"))
        end
        eval(Meta.parse("include(include_file)"))
    end
    nothing
end

function execute_trial(function_name::AbstractString, trial::Trial)::Tuple{UUID,Dict{Symbol,Any}}
    fn = eval(Meta.parse("$function_name"))
    results = fn(; trial.configuration...)
    return (trial.id, results)
end

function execute_trial_and_save_to_db_async(function_name::AbstractString, trial::Trial)
    (id, results) = execute_trial(function_name, trial)
    remotecall_wait(complete_trial_in_global_database, 1, (id, results))
    nothing
end

function set_global_database(db::ExperimentDatabase)
    global global_experiment_database = db
end

function complete_trial_in_global_database(trial_id::UUID, results::Dict{Symbol,Any})
    global global_experiment_database

    complete_trial!(global_experiment_database, trial_id, results)
    nothing
end

function run_trials(runner::Runner, trials::AbstractArray{Trial})
    if runner == DistributedMode
        @info "Running $(length(trials)) trials across $(length(workers())) workers"
        set_global_database(db)
        configurations = (x -> x.configuration).(trials)
        function_names = (_ -> runner.experiment.function_name).(trials)
        pmap(execute_trial_and_save_to_db_async, workers(), function_names, configurations)
    elseif runner.execution_mode == MultithreadedMode
        @info "Running $(length(trials)) trials across $(Threads.nthreads()) threads"
        Threads.@threads for trial in trials
            (id, results) = execute_trial(runner.experiment.function_name, trial)
            complete_trial!(runner.database, id, results)
        end
    else
        @info "Running $(length(trials)) trials"
        for trial in trials
            (id, results) = execute_trial(runner.experiment.function_name, trial)
            complete_trial!(runner.database, id, results)
        end
    end
    @info "Finished all trials."
end

