import yaml


def load_config(path: str) -> dict:
    """
    Loads configuration from yaml file
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def extract_model_config(cfg: dict) -> dict:
    """
    Extract only model configuration essentials from config file.
    """
    return {
        "pca_components": cfg["svm"]["pca_components"],
        "svm_c": cfg["svm"]["c"],
        "svm_kernel": cfg["svm"]["kernel"],
        "svm_gamma": cfg["svm"]["gamma"],
        "w_svm": cfg["ensemble"]["w_svm"],
        "tau": cfg["ensemble"]["tau"],
        "xgb_n_estimators": cfg["xgb"]["n_estimators"],
        "xgb_max_depth": cfg["xgb"]["max_depth"],
        "xgb_learning_rate": cfg["xgb"]["learning_rate"],
        "xgb_subsample": cfg["xgb"]["subsample"],
        "xgb_colsample_bytree": cfg["xgb"]["colsample_bytree"],
        "random_seed": cfg["training"]["random_seed"],
    }
