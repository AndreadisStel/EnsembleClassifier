import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import os


def save_confusion_matrix(y_true, y_pred, out_dir="artifacts"):
    """
    Save confusion matrix.
    """

    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    path = os.path.join(out_dir, "confusion_matrix.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def evaluate(model, X_val, y_val, print_metrics=False, show_cm=False):

    """
    Returns accuracy metrics and confusion matrix path for logging.
    Optionally print metrics and display confusion matrix.
    """

    y_pred, assistant_activation_rate, assistant_change_rate = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    mask_hard = y_val.isin([2, 5])
    acc_hard = accuracy_score(y_val[mask_hard], y_pred[mask_hard])
    mask_easy = y_val.isin([1, 3, 4])
    acc_easy = accuracy_score(y_val[mask_easy], y_pred[mask_easy]) if mask_easy.sum() > 0 else 0

    if print_metrics:
        print("\n" + "="*50)
        print("Ensemble Model Evaluation")
        print("="*50)
        print(f"Overall Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
        print(f"Hard Accuracy (2 vs 5):   {acc_hard:.4f}")
        print(f"Easy Accuracy (1,3,4):    {acc_easy:.4f}")
        print("="*50)

    if show_cm:
        cm = confusion_matrix(y_val, y_pred)
        labels = sorted(y_val.unique())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
    cm_path = save_confusion_matrix(y_val, y_pred)

    metrics = {
        "global_accuracy": acc,
        "hard_accuracy": acc_hard,
        "easy_accuracy": acc_easy,
        "assistant_activation_rate": assistant_activation_rate,
        "assistant_change_rate": assistant_change_rate
    }

    return metrics, cm_path


def stress_test(model_class, X, y, config, n_folds=5):
    """
    Repeated CV stress test.
    Trains fresh models per fold.
    """

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=config["random_seed"],
    )

    accuracies = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = model_class(**config)
        model.fit(X_tr, y_tr)

        preds, _, _ = model.predict(X_va)
        acc = accuracy_score(y_va, preds)

        accuracies.append(acc)

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    # save summary artifact
    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/stress_test_summary.txt"

    with open(path, "w") as f:
        f.write(f"Mean accuracy: {mean_acc:.4f}\n")
        f.write(f"Std accuracy:  {std_acc:.4f}\n")
        f.write(f"Worst fold:    {min(accuracies):.4f}\n")
        f.write(f"Best fold:     {max(accuracies):.4f}\n")

    return mean_acc, std_acc, path
