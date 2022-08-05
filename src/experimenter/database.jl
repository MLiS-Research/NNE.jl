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
    INSERT OR IGNORE INTO Experiments (id, name, include_file, function_name, configuration, num_trials) VALUES (?, ?, ?, ?, ?, ?)
    """
    return SQLite.Stmt(db, sql)
end
function get_trial_insert_stmt(db::SQLite.DB)
    sql = raw"""
    INSERT OR IGNORE INTO Trials (id, experiment_id, configuration, results, trial_index, has_finished) VALUES (?, ?, ?, ?, ?, ?)
    """
    return SQLite.Stmt(db, sql)
end
function Base.push!(db::ExperimentDatabase, experiment::Experiment)
    vs = (string(experiment.id), experiment.name, experiment.include_file, experiment.function_name, experiment.configuration, experiment.num_trials)
    SQLite.execute(db._experimentInsertStmt, vs)
    nothing
end
function Base.push!(db::ExperimentDatabase, trial::Trial)
    vs = (string(trial.id), string(trial.experiment_id), trial.configuration, trial.results, trial.trial_index, trial.has_finished)
    SQLite.execute(db._trialInsertStmt, vs)
    nothing
end


Experiment(row::DataFrameRow) = Experiment(UUID(row.id), row.name, row.include_file, row.function_name, row.configuration, row.num_trials)
Trial(row::DataFrameRow) = Trial(
    id=UUID(row.id),
    experiment_id=UUID(row.experiment_id),
    configuration=row.configuration,
    results=row.results,
    trial_index=row.trial_index,
    has_finished=row.has_finished
)

function prepare_db(db::SQLite.DB)
    # Create a table for the experiments
    experiments_query = raw"""
    CREATE TABLE IF NOT EXISTS Experiments (
        id TEXT NOT NULL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        include_file TEXT,
        function_name TEXT,
        configuration BLOB,
        num_trials INTEGER NOT NULL
    );
    """

    trials_query = raw"""
    CREATE TABLE IF NOT EXISTS Trials (
        id TEXT NOT NULL PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        configuration BLOB,
        results BLOB,
        trial_index INTEGER NOT NULL,
        has_finished BOOLEAN NOT NULL, 
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


function get_experiment(db::ExperimentDatabase, experiment_id)
    experiment_id = SQLite.esc_id(string(experiment_id))
    df = (SQLite.DBInterface.execute(db._db, "SELECT * FROM Experiments WHERE id = $experiment_id") |> DataFrame)
    return Experiment(first(eachrow(df)))
end

function get_experiment_by_name(db::ExperimentDatabase, name)
    name = SQLite.esc_id(string(name))
    df = (SQLite.DBInterface.execute(db._db, "SELECT * FROM Experiments WHERE name = $name") |> DataFrame)
    return Experiment(first(eachrow(df)))
end


function check_overlap(experimentA::Experiment, experimentB::Experiment)
    if experimentA.name != experimentB.name
        return false
    elseif experimentA.function_name != experimentB.function_name
        return false
    elseif experimentA.num_trials != experimentB.num_trials
        return false
    else
        trials_a = collect(experimentA)
        trials_b = collect(experimentB)

        for (a, b) in zip(trials_a, trials_b)
            if a.configuration != b.configuration
                return false
            end
        end
    end

    return true
end

function restore_from_db(db::ExperimentDatabase, experiment::Experiment)
    name = SQLite.esc_id(string(experiment.name))
    df = (SQLite.DBInterface.execute(db._db, "SELECT * FROM Experiments WHERE name = $name") |> DataFrame)
    if length(eachrow(df)) > 0
        existing_experiment = Experiment(first(eachrow(df)))
        if (!check_overlap(experiment, existing_experiment))
            error("Found existing experiment with name \"$(experiment.name)\", but with different parameters. Use a different name.")
        end
        return existing_experiment
    end

    return experiment
end

function get_experiments(db::ExperimentDatabase)
    df = SQLite.DBInterface.execute(db._db, "SELECT * FROM Experiments") |> DataFrame
    return [Experiment(row) for row in eachrow(df)]
end

function get_trial(db::ExperimentDatabase, trial_index)
    trial_index = SQLite.esc_id(string(trial_index))
    df = (SQLite.DBInterface.execute(db._db, "SELECT * FROM Trials WHERE id = $trial_index") |> DataFrame)
    return Trial(first(eachrow(df)))
end

function get_trials(db::ExperimentDatabase, experiment_id)
    experiment_id = SQLite.esc_id(string(experiment_id))
    df = SQLite.DBInterface.execute(db._db, "SELECT * FROM Trials WHERE experiment_id = $experiment_id ORDER BY trial_index ASC") |> DataFrame
    return [Trial(row) for row in eachrow(df)]
end

function get_trials_by_name(db::ExperimentDatabase, name)
    sql = raw"""
    SELECT name, Trials.id as id, experiment_id, Trials.configuration as configuration, results, trial_index, has_finished 
    FROM Trials 
    INNER JOIN Experiments ON Experiments.id == Trials.experiment_id 
    WHERE name = ? 
    ORDER BY trial_index
    """
    df = (SQLite.DBInterface.execute(db._db, sql, (name,)) |> DataFrame)
    return [Trial(row) for row in eachrow(df)]
end

function complete_trial!(db::ExperimentDatabase, trial_id::UUID, results::Dict{Symbol,Any})
    stmt = SQLite.Stmt(db._db, "UPDATE Trials SET results = @results, has_finished = @finished WHERE id = @id")
    vs = Dict{Symbol,Any}(:results => results, :finished => true, :id => string(trial_id))
    DBInterface.execute(stmt, vs)
    nothing
end
