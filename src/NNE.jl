module NNE

include("helpers/runner.jl")
include("linear_perceptron/exact_calculation.jl")
include("linear_perceptron/tps_calculation.jl")

include("mnist/mnist.jl") # MNIST module
include("toy_example/toy.jl")

end