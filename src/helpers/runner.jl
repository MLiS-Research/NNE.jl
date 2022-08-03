module Runner
using Distributed
using ProgressMeter
using Base.Iterators

@enum TaskExecutionMode SerialMode ThreadedMode DistributedMode

function get_results_distributed(fn, iter; show_progress=false)
    if show_progress
        results = @showprogress pmap(iter) do x
            fn(x...)
        end
        return results
    else
        results = pmap(iter) do x
            fn(x...)
        end
        return results
    end
end
function get_results_threaded(fn, iter; show_progress=false)
    results = Array{Any}(undef, size(iter)...)
    progress_bar = show_progress ? Progress(length(iter)) : nothing
    enumerated_iter = collect(enumerate(iter))
    Threads.@threads for (i, x) in enumerated_iter
        results[i] = fn(x...)
        show_progress && next!(progress_bar)
    end
    return results
end
function get_results_serial(fn, iter; show_progress=false)
    if show_progress
        results = @showprogress map(iter) do x
            fn(x...)
        end
        return results
    else
        results = map(iter) do x
            fn(x...)
        end
        return results
    end
end

function get_results(fn, iter, mode::TaskExecutionMode=SerialMode; show_progress=false)
    if mode==SerialMode
        return get_results_serial(fn, iter; show_progress)
    elseif mode==ThreadedMode
        return get_results_threaded(fn, iter; show_progress)
    elseif mode==DistributedMode
        return get_results_distributed(fn, iter; show_progress)
    else
        throw("TaskExecutionMode $mode is unimplemented")
    end
end

export TaskExecutionMode, get_results, SerialMode, ThreadedMode, DistributedMode

end