module CallbackGenerator
using TPS
using TPS.Callbacks
using TPS.SimulatedAnnealing
using TPS.DiscreteTrajectory
using ..Experimenter
using UUIDs
import Flux: cpu, gpu
using Logging
using Random

export create_callbacks, restore_state!

restore_state!(problem, ::Nothing; kwargs...) = nothing
function restore_state!(problem::DTProblem, restore_trial_id::UUID; kwargs...)
    results = get_results_from_trial_global_database(restore_trial_id)
    if ismissing(results)
        return nothing
    end

    final_states = results[:final_state]

    for (initial_state, final_state) in zip(problem.states, final_states)
        copy!(initial_state, final_state)
    end
    nothing
end
function restore_state!(problem::SAProblem, restore_trial_id::UUID; kwargs...)
    results = get_results_from_trial_global_database(restore_trial_id)
    if ismissing(results)
        return nothing
    end

    final_state = results[:final_state]

    copy!(problem.state, final_state)
    nothing
end

function sanitise_object(obj; skip_fields=Set{Symbol}())
    sanitised_obj = Dict{Symbol,Any}()
    for field in fieldnames(typeof(obj))
        if field in skip_fields
            continue
        end
        val = getfield(obj, field)
        if typeof(val) <: AbstractArray
            sanitised_obj[field] = val |> cpu
        elseif typeof(val) <: AbstractObservable
            continue
        elseif typeof(val) <: Function
            continue
        else
            sanitised_obj[field] = val
        end
    end
    return sanitised_obj
end
function override_object!(obj, santised_obj::Dict{Symbol,Any}; skip_fields=Set{Symbol}(), fn=identity, apply_fn_to_fields=Set{Symbol}())
    obj_fields = fieldnames(typeof(obj))
    for (field, val) in santised_obj
        if field in skip_fields
            continue
        end
        if !(field in obj_fields)
            error("Expected the field $field to be in $obj, but it is not.")
        end

        if field in apply_fn_to_fields
            setfield!(obj, field, val |> fn)
        elseif typeof(val) <: AbstractArray
            getfield(obj, field) .= val # Preserve types
        else
            setfield!(obj, field, val)
        end
    end
end

function move_state(states::AbstractArray{T}, device) where {T<:AbstractArray}
    return [state |> device for state in states]
end
function move_state(state, device)
    return state |> device
end

create_callbacks(::Nothing, device; kwargs...) = nothing
function create_callbacks(trial_id::UUID, device, info=nothing; save_final_snapshot::Bool=false, use_previous_snapshot::Bool=false, alternate_trial_id=nothing, snapshot_every_n=nothing, snapshot_label=missing, kwargs...)
    cb_storage = Dict{Symbol,Any}()
    can_restore = false
    if use_previous_snapshot
        snapshot = get_latest_snapshot_from_global_database(isnothing(alternate_trial_id) ? trial_id : alternate_trial_id)
        if !isnothing(snapshot)
            @debug "Using snapshot $(snapshot.id) from trial $(snapshot.trial_id)."
            cb_storage = snapshot.state
            can_restore = true
        end
    end

    if !isnothing(info)
        info[:acceptances] = Bool[]
    end
    # Fields to skip in snapshots
    skip_fields = Set((:exclude_parameter_mask, :indices_changed, :observable))

    function take_snapshot(deps::SolveDependencies)
        cb_storage[:current_state] = move_state(TPS.get_current_state(deps.solution), cpu)
        if isa(deps.solution, TPS.SimpleSolution)
            cb_storage[:observations] = deps.solution.observations
        end
        cb_storage[:cache] = sanitise_object(deps.cache; skip_fields)
        rng_state = copy(Random.default_rng())
        cb_storage[:rng_state] = rng_state
        if !isnothing(info)
            cb_storage[:acceptances] = info[:acceptances]
        end
        save_snapshot_in_global_database(trial_id, cb_storage, snapshot_label)
        nothing
    end
    function restore_snapshot(deps::SolveDependencies)
        # Short circuit if cannot restore
        if !can_restore
            return nothing
        end
        # restore rng
        if haskey(cb_storage, :rng_state)
            copy!(Random.default_rng(), cb_storage[:rng_state])
        else
            @debug "Did not find :rng_state in the snapshot state"
        end
        TPS.set_current_state!(deps.solution, move_state(cb_storage[:current_state], device))
        if isa(deps.solution, TPS.SimpleSolution)
            if haskey(cb_storage, :observations)
                deps.solution.observations = cb_storage[:observations]
            else
                @debug "Did not find :observations in the snapshot state"
            end
        end
        apply_fn_to_fields = Set((:state, :state_cache))
        override_object!(deps.cache, cb_storage[:cache]; fn=(x -> move_state(x, device)), apply_fn_to_fields, skip_fields)
        if !isnothing(info)
            if haskey(cb_storage, :acceptances)
                info[:acceptances] = cb_storage[:acceptances]
            else
                @debug "Did not find :acceptances in the snapshot state"
            end
        end
        nothing
    end
    function record_acceptances(deps::SolveDependencies)
        push!(info[:acceptances], TPS.MetropolisHastings.last_accepted(deps.cache))
    end
    function throttle(fn, n)
        counter = 0
        function execute(args...; kwargs...)
            counter += 1
            if counter % n == 0
                fn(args...; kwargs...)
            end
        end
        return execute
    end

    callbacks = []
    if use_previous_snapshot
        push!(callbacks, InitialisationCallback(restore_snapshot))
    end
    if !isnothing(info)
        push!(callbacks, PostInnerLoopCallback(record_acceptances))
    end
    if !isnothing(snapshot_every_n)
        push!(callbacks, PostInnerLoopCallback(throttle(take_snapshot, snapshot_every_n)))
    end
    if save_final_snapshot
        push!(callbacks, FinalisationCallback(take_snapshot))
    end

    if length(callbacks) == 0
        return nothing
    elseif length(callbacks) == 1
        return callbacks[begin]
    else
        return CallbackSet(callbacks...)
    end
end
end