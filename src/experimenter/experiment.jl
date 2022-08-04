using Base
using Base.Iterators
using UUIDs

abstract type AbstractVariable end
"""
    count_values(entry)

Counts the number of different values which 'variable' can take.
"""
count_values(variable) = 1
count_values(variable::AbstractVariable) = error("count_values is not defined for $(typeof(variable)).")

"""
    extract_value(variable, i)

Returns the i^th possible value of 'variable'. 
'i' follows 1 based indexing.
"""
extract_value(variable, i) = v
extract_value(variable::AbstractVariable, i) = error("extract_value is not defined for $(typeof(variable)).")


function checkbounds(v::AbstractVariable, i)
    (i < 1 || i > count_values(v)) && error("Cannot access $(typeof(v)) with $(count_values(v)) elements at index $i").
    nothing
end

# AbstractVariables should implement the iteration interfaces.
Base.iterate(v::AbstractVariable) = (extract_value(v, 1), 2)
function Base.iterate(v::AbstractVariable, i)
    if i <= length(v)
        return (extract_value(v, i), i + 1)
    else
        return nothing
    end
end
Base.length(v::AbstractVariable) = count_values(v)

struct LinearVariable{T,Q<:Integer} <: AbstractVariable
    min_value::T
    max_value::T
    num_values::Q
end
count_values(v::LinearVariable) = v.num_values
function extract_value(v::LinearVariable, i)
    checkbounds(v, i)
    val = v.min_value + (v.max_value - v.min_value) * (i - 1) / (v.num_values - 1)
    return val
end
Base.eltype(::LinearVariable{T}) where {T} = promote(T, Float64)


struct RepeatVariable{T,Q<:Integer} <: AbstractVariable
    value::T
    num_repeats::Q
end
count_values(v::RepeatVariable) = v.num_repeats
function extract_value(v::RepeatVariable, i)
    checkbounds(v, i)
    return v.value
end
Base.eltype(::RepeatVariable{T}) where {T} = T

struct LogLinearVariable{T,Q<:Integer} <: AbstractVariable
    min_value::T
    max_value::T
    num_values::Q
end
count_values(v::LogLinearVariable) = v.num_values
function extract_value(v::LogLinearVariable{T}, i) where {T}
    checkbounds(v, i)
    if i == 1
        return v.min_value
    elseif i == v.num_values
        return v.max_value
    end
    log_min_value = log10(v.min_value)
    log_max_value = log10(v.max_value)
    return convert(Float64, 10.0 .^ (log_min_value + (log_max_value - log_min_value) * (i - 1) / (v.num_values - 1)))
end
Base.eltype(::LogLinearVariable{T}) where {T} = promote(Float64, T)

struct IterableVariable{Q,T<:AbstractArray{Q}} <: AbstractVariable
    iterator::T
end
count_values(v::IterableVariable) = length(v.iterator)
Base.eltype(::LogLinearVariable{Q,T}) where {Q,T} = Q
Base.iterate(v::IterableVariable) = iterate(v.iterator)
Base.iterate(v::IterableVariable, state) = iterate(v.iterator, state)
extract_value(v::IterableVariable, i) = getindex(v.iterator, i)

Base.@kwdef struct Experiment
    id::UUID = uuid4()
    include_file::AbstractString
    code::AbstractString
    configuration::Dict{Symbol,Any}
end

Base.@kwdef struct Trial
    id::UUID = uuid4()
    experiment_id::UUID
    configuration::Dict{Symbol,Any}
end

function count_trails(experiment::Experiment)
    return mapreduce(count_values, *, values(experiment.configuration))
end

function _construct_trial(id::UUID, experiment::Experiment, param_map)
    config_dict = Dict{Symbol,Any}()

    for (key, value) in experiment.configuration
        if haskey(param_map, key)
            config_dict[key] = param_map[key]
        else
            config_dict[key] = value
        end
    end

    return Trial(id=id, configuration=config_dict, experiment_id=experiment.id)
end

function combinatorial_iterator(config)
    return product((Iterators.map((v_i) -> Dict(sym => v_i), v) for (sym, v) in config if typeof(v) <: AbstractVariable)...)
end

function getrng(id::UUID)
    seed = id.value
    return UUIDs.Random.MersenneTwister(seed)
end

function Base.iterate(experiment::Experiment)
    isnothing(experiment.configuration) && return nothing

    config = experiment.configuration
    iter = combinatorial_iterator(config)

    rng = getrng(experiment.id)

    if (length(iter) == 0)
        return nothing
    end

    param_map_tuple, iter_state = iterate(iter)
    param_map = merge(param_map_tuple...)

    trial = _construct_trial(uuid4(rng), experiment, param_map)

    next_state = (iter, iter_state, rng)

    return trial, next_state
end

function Base.iterate(experiment::Experiment, state)
    (iter, last_state, rng) = state
    coll_iter = iterate(iter, last_state)
    if isnothing(coll_iter)
        return nothing
    end

    param_map_tuple, iter_state = coll_iter
    param_map = merge(param_map_tuple...)

    trial = _construct_trial(uuid4(rng), experiment, param_map)

    next_state = (iter, iter_state, rng)
    return trial, next_state
end

Base.length(experiment::Experiment) = count_trails(experiment)
Base.eltype(::Experiment) = Trial