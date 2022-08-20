-- SQLite
SELECT name, CAST(sum(has_finished) as REAL) / COUNT(*) * 100 as percent_finished FROM (SELECT name, trial_index, has_finished
FROM Trials 
INNER JOIN Experiments ON Trials.experiment_id = Experiments.id
) GROUP BY name;