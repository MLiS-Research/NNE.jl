using NNE
using NNE.Experimenter

primary_db = open_db("experiments.db", joinpath(pwd(), "results", "large"))
secondary_db = open_db("experiments_secondary.db", joinpath(pwd(), "results", "large"))


merge_databases!(primary_db, secondary_db)