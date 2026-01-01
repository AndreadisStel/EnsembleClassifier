import csv
import os
from datetime import datetime


LOG_PATH = "artifacts/inference_logs.csv"


def log_inference(svm_margin, prediction, assistant_used):
    os.makedirs("artifacts", exist_ok=True)

    write_header = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "timestamp",
                "prediction",
                "svm_margin",
                "assistant_used"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            prediction,
            svm_margin,
            assistant_used
        ])
