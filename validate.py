"""
Compares the latest trained model against the best model seen so far.
Returns True if the new model is better and should be saved, False otherwise.
"""

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_runs(client, experiment_name):
    # TODO fetch all runs for the experiment,
    # ordered by start_time (most recent first).
    # Raise a RuntimeError if no runs are found.
    # Return the list of runs.
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found in the experiment.")
    return runs


def main():
    params = load_params()
    mf = params["mlflow"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()

    runs = get_runs(client, mf["experiment_name"])

    # TODO Get the val_accuracy of the most recent run (runs[0]).

    accuracy=runs[0].data.metrics["val_accuracy"]
    # TODO Find the best val_accuracy across all previous runs.
    # Hint: use max() with a generator expression.
    # Hint: Consider the case where there is only one run.

    # TODO Return True if the new model strictly improves on the best
    #      previous accuracy, False otherwise
    best_previous = max(
        (run.data.metrics["val_accuracy"] for run in runs[1:]),
        default=None
    )

    return best_previous is None or accuracy > best_previous


if __name__ == "__main__":
    main()
