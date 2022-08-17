using NNE
using NNE.Experimenter

db = open_db("export_sulis_mnist_results.db", joinpath(pwd(), "results", "large"))

trials = get_trials_by_name(db, "MNIST Final Results 1");

