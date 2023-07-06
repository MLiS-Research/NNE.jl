module NNE

include("helpers/problem_generators.jl")
include("helpers/runner.jl")


include("linear_perceptron/exact_calculation.jl")
include("linear_perceptron/tps_calculation.jl")
include("classification/toy_problem.jl")

include("helpers/callback_generator.jl")

include("mnist/mnist.jl") # MNIST module
include("toy_example/toy.jl")

end