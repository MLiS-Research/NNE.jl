using Distributed, CUDA
addprocs(length(devices()))
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using CUDA

# assign devices
function remote_map_to_device(p, d)
    return remotecall(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end

futures = map((x) -> remote_map_to_device(x...), (zip(workers(), devices())))

# Wait for all of the future
wait.(futures)