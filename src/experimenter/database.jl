using SQLite
using Logging
using Base
using UUIDs
using DataFrames

struct ExperimentDatabase
    experiment_folder::AbstractString
    database_name::AbstractString
    _db::SQLite.DB
    _experimentInsertStmt::SQLite.Stmt
    _trialInsertStmt::SQLite.Stmt
end

function get_experiment_insert_stmt(db::SQLite.DB)
    sql = raw"""
    INSERT INTO Experiments (id, include_file, code, configuration) VALUES (?, ?, ?, ?);
    """
    return SQLite.Stmt(db, sql)
end
function get_trial_insert_stmt(db::SQLite.DB)
    sql = raw"""
    INSERT INTO Trials (id, experiment_id, configuration) VALUES (?, ?, ?);
    """
    return SQLite.Stmt(db, sql)
end


function prepare_db(db::SQLite.DB)
    # Create a table for the experiments
    experiments_query = raw"""
    CREATE TABLE IF NOT EXISTS Experiments (
        id TEXT NOT NULL PRIMARY KEY,
        include_file TEXT,
        code TEXT,
        configuration BLOB
    );
    """

    trials_query = raw"""
    CREATE TABLE IF NOT EXISTS Trials (
        id TEXT NOT NULL PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        configuration BLOB,
        FOREIGN KEY (experiment_id) REFERENCES Experiments (id)
            ON DELETE CASCADE ON UPDATE CASCADE
    );
    """

    # Allow foreign keys
    SQLite.execute(db, "PRAGMA foreign_keys = ON;")

    SQLite.execute(db, experiments_query)
    SQLite.execute(db, trials_query)
    nothing
end

function open_db(database_name, experiment_folder=joinpath(pwd(), "experiments"), create_folder=true)::ExperimentDatabase
    if (!Base.Filesystem.isdir(experiment_folder))
        if create_folder
            @info "Creating $experiment_folder for experiments folder."
            Base.Filesystem.mkdir(experiment_folder)
        else
            error_msg = "$experiment_folder does not exist for database, and create_folder is set to false."
            @error error_msg
            error(error_msg)
        end
    end
    _sqliteDB = SQLite.DB(joinpath(experiment_folder, database_name))
    prepare_db(_sqliteDB)
    experiment_stmt = get_experiment_insert_stmt(_sqliteDB)
    trial_stmt = get_trial_insert_stmt(_sqliteDB)
    db = ExperimentDatabase(experiment_folder, database_name, _sqliteDB, experiment_stmt, trial_stmt)

    return db
end

function Base.push!(db::ExperimentDatabase, experiment::Experiment)
    vs = (string(experiment.id), experiment.include_file, experiment.code, experiment.configuration)
    SQLite.execute(db._experimentInsertStmt, vs)
    nothing
end

function Base.push!(db::ExperimentDatabase, trial::Trial)
    vs = (string(trial.id), string(trial.experiment_id), trial.configuration)
    SQLite.execute(db._trialInsertStmt, vs)
    nothing
end

Experiment(row::DataFrameRow) = Experiment(UUID(row.id), row.include_file, row.code, row.configuration)
Trial(row::DataFrameRow) = Trial(UUID(row.id), UUID(row.experiment_id), row.configuration)

function get_experiment(db::ExperimentDatabase, experiment_id)
    experiment_id = SQLite.esc_id(string(experiment_id))
    vs = (SQLite.DBInterface.execute(db._db, "SELECT * FROM Experiments WHERE id = $experiment_id") |> DataFrame)
    return Experiment(first(eachrow(vs)))
end

function get_experiments(db::ExperimentDatabase)
    df = SQLite.DBInterface.execute(db._db, "SELECT * FROM Experiments") |> DataFrame
    return [Experiment(row) for row in eachrow(df)]
end

function get_trial(db::ExperimentDatabase, trial_id)
    trial_id = SQLite.esc_id(string(trial_id))
    vs = (SQLite.DBInterface.execute(db._db, "SELECT * FROM Trials WHERE id = $trial_id") |> DataFrame)
    return Trial(first(eachrow(vs)))
end

function get_trials(db::ExperimentDatabase, experiment_id)
    experiment_id = SQLite.esc_id(string(experiment_id))
    df = SQLite.DBInterface.execute(db._db, "SELECT * FROM Trials WHERE experiment_id = $experiment_id") |> DataFrame
    return [Trial(row) for row in eachrow(df)]
end
