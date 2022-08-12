using Base
using Distributed
using Base.Iterators
using Logging
using ProgressBars
using Pkg

@enum EXECUTEMODE SerialMode MultithreadedMode DistributedMode

Base.@kwdef struct Runner
    execution_mode::EXECUTEMODE
    experiment::Experiment
    database::ExperimentDatabase
end

macro execute(experiment, database, mode=SerialMode, use_progress=false, directory=pwd())
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

            if runner.execution_mode == DistributedMode
                current_environment = dirname(Pkg.project().path)
                dir = $(esc(directory))
                @everywhere using Pkg
                eval(Meta.parse("@everywhere Pkg.activate(raw\"$(current_environment)\")"))
                @everywhere using NNE.Experimenter
                # Make sure each worker is in the right directory
                eval(Meta.parse("@everywhere cd(raw\"$(dir)\")"))
            end


            include_file = runner.experiment.include_file
            if !ismissing(include_file)
                if runner.execution_mode == DistributedMode
                    eval(Meta.parse("@everywhere include(raw\"$include_file\");"))
                end
                eval(Meta.parse("include(raw\"$include_file\")";))
            end


            run_trials(runner, incomplete_trials; use_progress=$(esc(use_progress)))
        end
    end
end

function execute_trial(function_name::AbstractString, trial::Trial)::Tuple{UUID,Dict{Symbol,Any}}
    fn = eval(Meta.parse("$function_name"))
    results = fn(trial.configuration, trial.id)
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
function unset_global_database()
    global global_experiment_database = nothing
end

function complete_trial_in_global_database(trial_id::UUID, results::Dict{Symbol,Any})
    global global_experiment_database

    complete_trial!(global_experiment_database, trial_id, results)
    nothing
end

function save_snapshot_in_global_database(trial_id::UUID, state::Dict{Symbol, Any}, label=missing)
    # Redirect requests on worker nodes to the main node
    if myid()!=1
        remotecall_wait(save_snapshot_in_global_database, 1, (trial_id, state, label))
        return nothing
    end

    global global_experiment_database

    save_snapshot!(global_experiment_database, trial_id, state, label)
    nothing
end

function get_latest_snapshot_from_global_database(trial_id::UUID)
    # Redirect requests on worker nodes to main node
    if myid()!=1
        return remotecall_wait(get_latest_snapshot, 1, (trial_id))
    end

    global global_experiment_database
    return latest_snapshot(global_experiment_database, trial_id)
end

export get_latest_snapshot_from_global_database, save_snapshot_in_global_database

function run_trials(runner::Runner, trials::AbstractArray{Trial}; use_progress=false)
    if length(trials) == 0
        @info "No incomplete trials found. Finished."
        return nothing
    end

    iter = use_progress ? ProgressBar(trials) : trials
    if runner == DistributedMode && length(workers()) <= 1
        @info "Only one worker found, switching to serial execution."
        runner = SerialMode
    end
    set_global_database(runner.database)
    if runner == DistributedMode
        @info "Running $(length(trials)) trials across $(length(workers())) workers"
        configurations = (x -> x.configuration).(trials)
        function_names = (_ -> runner.experiment.function_name).(trials)
        use_progress && @debug "Progress bar not supported in distributed mode."
        pmap(execute_trial_and_save_to_db_async, function_names, configurations)
    elseif runner.execution_mode == MultithreadedMode
        @info "Running $(length(trials)) trials across $(Threads.nthreads()) threads"
        Threads.@threads for trial in iter
            (id, results) = execute_trial(runner.experiment.function_name, trial)
            complete_trial!(runner.database, id, results)
        end
    else
        @info "Running $(length(trials)) trials"
        for trial in iter
            (id, results) = execute_trial(runner.experiment.function_name, trial)
            complete_trial!(runner.database, id, results)
        end
    end
    unset_global_database()
    @info "Finished all trials."
    nothing
end

