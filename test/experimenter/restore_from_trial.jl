using NNE
using NNE.Experimenter
import NNE.Experimenter: open_db, @execute
using SafeTestsets
using Test
import Base.Iterators: product

function get_test_config()
    return Dict{Symbol,Any}(
        :s => IterableVariable([0.0, 0.1]),
        :τ => IterableVariable([1, 4]),
        :σ => 10.0,
        :d => 10,
        :epochs => 4
    )
end
function get_test_config(restore_from_experiment::Experiment)
    conf = get_test_config()
    conf[:restore_from_trial_id] = MatchIterableVariable([trial.id for trial in restore_from_experiment])
    return conf
end

function get_experiment(name, config)
    experiment = Experiment(
        name=name,
        include_file="trial_functions.jl",
        function_name="run_restore_experiment",
        configuration=config
    )
    return experiment
end

@testset "Restore from experiment" begin
    experiment = get_experiment("Initial trial", get_test_config())
    database = open_db("restore from trial test"; in_memory=true)

    file_path = @__FILE__
    directory = dirname(file_path)

    @execute experiment database SerialMode false directory

    restore_experiment = get_experiment("Second trial", get_test_config(experiment))

    @execute restore_experiment database SerialMode false directory

    first_trials = get_trials(database, experiment.id)
    restored_trials = get_trials(database, restore_experiment.id)

    @test length(first_trials) == length(restored_trials)

    for (original_trial, restored_trial) in zip(first_trials, restored_trials)
        @test !ismissing(original_trial.results)
        @test !ismissing(restored_trial.results)

        @test all(original_trial.results[:final_state] .== restored_trial.results[:initial_state])
    end
end