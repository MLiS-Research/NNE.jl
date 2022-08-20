using Plots
using DataFrames
using Statistics
using Dates
using Plots.Measures
using NNE
using NNE.Experimenter
using SQLite
using Dates

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
to_seconds(x) = getfield(convert(Dates.Millisecond, x), :value) / 1000.0
function plot_execution_map(trials, colour_key=nothing; colour_scheme=:matter)
    df = DataFrame([t for t in trials if t.has_finished])

    worker_ids = (x->x[:worker_id]).(df.results)
    start_times = (x->x[:start_time]).(df.results)
    end_times = (x->x[:end_time]).(df.results)
    colours = (x->nothing).(df.results)

    if !isnothing(colour_key)
        colour_codes = (x-> x[colour_key]).(df.results)
        unique_colours_map = Dict(c=>i for (i, c) in enumerate(sort(collect(Set(colour_codes)))))
        col_scheme = cgrad(colour_scheme, length(unique_colours_map), categorical = true)
        colours = (x->col_scheme[unique_colours_map[x]]).(colour_codes)
    end

    experiment_start = minimum(start_times)
    offset_start_times = to_seconds.(start_times .- experiment_start)
    offset_end_times = to_seconds.(end_times .- experiment_start)
    
    plt = plot(;)
    for (idx, start_time, end_time, c) in zip(worker_ids, offset_start_times, offset_end_times, colours)
        if isnothing(c)
            plot!(plt, rectangle(end_time-start_time, 1, start_time, idx))
        else
            plot!(plt, rectangle(end_time-start_time, 1, start_time, idx); color=c, colorbar=true, colorbartitle=string(colour_key))
        end
    end
    plot!(plt; legend=false)
    xlabel!(plt, "Time (s)")
    ylabel!(plt, "Trial Index")

    if !isnothing(colour_key)
        plot!(plt; colorbar=true, colorbartitle=string(colour_key), figsize=(1200,1200), dpi=300)
    end
    return plt
end

function plot_snapshot_map(db::NNE.Experimenter.ExperimentDatabase, trials, colour_key=nothing; colour_scheme=:matter)
    df = DataFrame([t for t in trials if t.has_finished])

    start_times = (x->x[:start_time]).(df.results)
    end_times = (x->x[:end_time]).(df.results)

    experiment_start = minimum(start_times)
    offset_start_times = to_seconds.(start_times .- experiment_start)
    offset_end_times = to_seconds.(end_times .- experiment_start)
    colours = (x->nothing).(df.results)

    if !isnothing(colour_key)
        colour_codes = (x-> x[colour_key]).(df.results)
        unique_colours_map = Dict(c=>i for (i, c) in enumerate(sort(collect(Set(colour_codes)))))
        col_scheme = cgrad(colour_scheme, length(unique_colours_map), categorical = true)
        colours = (x->col_scheme[unique_colours_map[x]]).(colour_codes)
    end

    plt = plot(;)
    for (trial, start_time, end_time, c) in zip(trials, offset_start_times, offset_end_times, colours)
        trial_id = SQLite.esc_id(string(trial.id))
        snapshot_dates = SQLite.DBInterface.execute(db._db, "SELECT created_at FROM Snapshots WHERE trial_id = $trial_id ORDER BY created_at ASC") |> DataFrame
        snapshot_dates = (x->DateTime(x, DateFormat("y-m-d H:M:S.s"))).(collect(snapshot_dates.created_at))
        offset_snapshot_times = to_seconds.(snapshot_dates .- experiment_start)
        insert!(offset_snapshot_times, 1, start_time)
        push!(offset_snapshot_times, end_time)
        idx = trial.results[:worker_id]
        plot!(plt, rectangle(end_time-start_time, 1, start_time, idx); alpha=0.6, color = isnothing(c) ? :auto : c)
        for (s_t, e_t) in zip(offset_snapshot_times[1:end-1], offset_snapshot_times[2, end]) 
            plot!(plt, rectangle(e_t-s_t, 1, s_t, idx); alpha=0.4, color = isnothing(c) ? :auto : c)
        end
    end
    plot!(plt; legend=false)
    xlabel!(plt, "Time (s)")
    ylabel!(plt, "Trial Index")

    return plt
end