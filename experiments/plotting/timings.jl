using Plots
using DataFrames
using Statistics
using Dates

function plot_execution_map(trials)
    df = DataFrame([t for t in trials if t.has_finished])

    trial_indices = df.trial_index
    start_times = (x->x[:start_time]).(df.results)
    end_times = (x->x[:end_time]).(df.results)

    experiment_start = minimum(start_times)
    to_seconds(x) = getfield(convert(Dates.Millisecond, x), :value) / 1000.0
    offset_start_times = to_seconds.(start_times .- experiment_start)
    offset_end_times = to_seconds.(end_times .- experiment_start)
    
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    plt = plot(;)
    for (idx, start_time, end_time) in zip(trial_indices, offset_start_times, offset_end_times)
        plot!(plt, rectangle(end_time-start_time, 1, start_time, idx))
    end
    plot!(plt; legend=false)
    xlabel!(plt, "Time (s)")
    ylabel!(plt, "Trial Index")

    return plt
end