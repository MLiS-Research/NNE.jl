using NNE
using NNE.Experimenter

db = open_db("experiments.db", joinpath(pwd(), "results", "large"))
export_db_file = joinpath(pwd(), "results", "large", "export.db")

export_db(db, export_db_file)