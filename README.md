# Neural Network Ensembles (NNEs) as realised through trajectory sampling techniques

[Development Board](https://github.com/JamieMair/nne-trajectory-sampling-code/projects/1)


## Setup

As this project uses `PyCall.jl`, this needs to be setup. Run the following:
```bash
julia --project
```
Then when in the REPL:
```julia
using Pkg
Pkg.instantiate()
ENV["PYTHON"] = "" # Force PyCall to use conda
Pkg.build("PyCall")
Pkg.precompile()
using Conda
Conda.pip("install", "tensorflow")
Conda.pip("install", "tensorboard")
```