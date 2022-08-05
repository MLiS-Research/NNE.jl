using Base
using Distributed
using Base.Iterators
using Logging

@enum EXECUTEMODE SerialMode MultithreadedMode DistributedMode

Base.@kwdef struct Runner
    execution_mode::EXECUTEMODE
    experiment::Experiment
    database::ExperimentDatabase
end

macro execute(experiment, database, mode=SerialMode)
    quote
        $(esc(experiment)) = restore_from_db($(esc(database)), $(esc(experiment)))
        let runner = Runner(experiment=$(esc(experiment)), database=$(esc(database)), execution_mode=$(esc(mode)))
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

            current_directory = pwd()
            if runner.execution_mode == DistributedMode
                @everywhere using Pkg
                @everywhere Pkg.activate(".")
                @everywhere using NNE.Experimenter
                # Make sure each worker is in the right directory
                eval(Meta.parse("@everywhere cd(\"$current_directory\");"))
            end


            include_file = runner.experiment.include_file
            if !ismissing(include_file)
                if runner.execution_mode == DistributedMode
                    eval(Meta.parse("@everywhere include(\"$include_file\");"))
                end
                eval(Meta.parse("include(\"$include_file\")";))
            end


            run_trials(runner, incomplete_trials)
        end
    end
end

function execute_trial(function_name::AbstractString, trial::Trial)::Tuple{UUID,Dict{Symbol,Any}}
    fn = eval(Meta.parse("$function_name"))
    results = fn(trial.configuration)
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
    if length(trials) == 0
        @info "No incomplete trials found. Finished."
        return nothing
    end

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
    nothing
end

