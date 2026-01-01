import mlflow
import joblib
from pathlib import Path
from src.model import EnsembleModel
from src.data import load_and_split_data, load_full_training_data
from src.evaluation import evaluate, stress_test
from src.utils import load_config, extract_model_config


def main():

    # 1. Load config
    cfg = load_config("config/config.yaml")
    model_cfg = extract_model_config(cfg)

    # 2. MLflow setup
    mlflow.set_experiment(cfg["training"]["experiment_name"])

    # 3. Run experiment
    with mlflow.start_run(run_name=cfg["training"]["run_name"]):

        # Log configuration
        mlflow.log_params(cfg)

        # Load data
        X_train, X_val, y_train, y_val = load_and_split_data(
            file_path=cfg["data"]["train_path"],
            test_size=cfg["data"]["validation_size"],
            random_state=cfg["training"]["random_seed"]
        )

        # Initialize & train model
        model = EnsembleModel(**model_cfg)
        model.fit(X_train, y_train)

        # Validation evaluation
        metrics, cm_path = evaluate(model, X_val, y_val)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)

        # Stress test
        X, y = load_full_training_data(file_path=cfg["data"]["train_path"])
        stress_mean, stress_std, stress_path = stress_test(
            EnsembleModel,
            X, y,
            model_cfg,
            n_folds=5
        )
        mlflow.log_metrics({
            "stress_test_mean_accuracy": stress_mean,
            "stress_test_std_accuracy": stress_std
        })
        mlflow.log_artifact(stress_path)

        # Save & log ensemble model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / "ensemble_model.pkl"
        joblib.dump(model, model_path)

        mlflow.log_artifact(str(model_path))


if __name__ == "__main__":
    main()
