using SQLite
using Logging

struct ExperimentDatabase
    experiment_folder::AbstractString
    database_name::AbstractString
    _db::SQLite.DB
end

function create_db(database_name, experiment_folder=joinpath(pwd(), "experiments"), create_folder=true)::ExperimentDatabase
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
    db = ExperimentDatabase(experiment_folder, database_name, _sqliteDB)
    return db
end


export create_db