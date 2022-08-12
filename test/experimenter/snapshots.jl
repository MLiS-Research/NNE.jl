using NNE
using NNE.Experimenter
using NNE.Experimenter: @execute
using Logging
using Distributed

Logging.disable_logging(Logging.Info)

function get_test_config()
    return Dict{Symbol,Any}(
        :s => 0.1,
        :τ => IterableVariable([1, 4]),
        :σ => 1.0,
        :epochs => 20,
    )
end

function get_experiment(name, config)
    experiment = Experiment(
        name=name,
        include_file="trial_toy_problem.jl",
        function_name="run_problem",
        configuration=config
    )
    return experiment
end

@testset "Create and restore snapshots" begin
    database = open_db("snapshots"; in_memory=true)
    experiment = get_experiment("Snapshot Test", get_test_config())

    @execute experiment database SerialMode

    trials = get_trials_by_name(database, experiment.name)
    @test length(trials)==2

    original_results = (x-> x.results).(trials)

    # Allow trials to restart
    for trial in trials
        mark_trial_as_incomplete!(database, trial.id)
    end

    @execute experiment database SerialMode

    new_trials = get_trials_by_name(database, experiment.name)

    for (new_trial, old_trial) in zip(new_trials, trials)
        @test new_trial.results != old_trial.results
        old_losses = old_trial.results[:observations]
        new_losses = new_trial.results[:observations]
        # New losses should have old ones included
        @test all(new_losses[1:length(old_losses)] .== old_losses)
        # As well as new losses
        @test length(new_losses) > length(old_losses)
    end
end