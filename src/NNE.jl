module NNE

include("interfaces.jl")
import .Interfaces
include("helpers/utils.jl")
import .Utils
include("ensembles.jl")

include("experiments/experiments.jl")

include("helpers/problem_generators.jl")
include("helpers/runner.jl")

include("linear_perceptron/exact_calculation.jl")
include("linear_perceptron/tps_calculation.jl")
include("classification/toy_problem.jl")

include("helpers/callback_generator.jl")

include("image_classification/image_classification.jl")
include("mnist/mnist.jl") # MNIST module
include("toy_example/toy.jl")

end