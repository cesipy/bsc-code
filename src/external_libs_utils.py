## This module includes utils for external libraries that need further tuning, b
# currenty includes optuna and mlflow

from logger import Logger

logger = Logger()


def get_or_create_experiment_id(name: str) -> str:
    """Resolve an *active* experiment id for `name`.

    Deleting an experiment in the MLflow UI only soft-deletes it: the name stays
    reserved with lifecycle_stage='deleted', and `mlflow.set_experiment(name)`
    then raises ("Cannot set a deleted experiment ... as the active experiment").
    This helper restores a soft-deleted experiment (or creates a fresh one)
    instead of crashing, so removing the project in the UI is non-fatal.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        return client.create_experiment(name)
    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        logger.info(f"restored soft-deleted MLflow experiment '{name}'")
    return exp.experiment_id


