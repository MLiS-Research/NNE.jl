function run_experiment(config, trial_id)
    info = Dict{Symbol, Any}()

    info[:value] = config[:n] * config[:m]
    info[:config] = config
    info[:trial_id] = trial_id
    
    return info
end