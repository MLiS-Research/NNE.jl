# Experiments

This is the directly containing all the entry points into the functions defined in the NNE.jl module.

## Creating a sysimage

In order to run the code quickly, it is best to precompile a sysimage, especially for use on a distributed system. This makes sure that the code can run very quickly, without needing to spend a lot of time precompiling.

First of all, open a REPL in this `experiments` directory, with the number of threads you want:
```bash
julia --project -t 8
```
Then, type the following:
```julia
using PackageCompiler
to_compile=["NNE", "TransitionPathSampling", "CUDA", "MLDatasets", "Flux", "Distributed", "SQLite", "LinearAlgebra", "Random", "Statistics"]
PackageCompiler.create_sysimage(to_compile; sysimage_path="julia_sysimage.so", precompile_statements_file="precompile.jl")
```

Then you can launch this sysimage file with:
```bash
julia --sys-image=julia_sysimage.so
```