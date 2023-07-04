using NNE
using NNE.Experimenter
db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

trials = get_trials_by_name(db, "MNIST");

should_prepare = true
if should_prepare
    include("plotting/mnist_plotting.jl");
    prepare_mnist_results(trials, max_loss_samples=Int(5e6),device=gpu,outputs=10,use_progress=true)
end