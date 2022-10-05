using NNE
using NNE.Experimenter

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))

trials = get_trials_by_name(db, "MNIST");

