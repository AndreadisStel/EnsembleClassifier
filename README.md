# Confidence-Gated Ensemble Classifier

This repository presents a machine learning system
built around a confidence-gated ensemble architecture.

Originally designed as a Machine Learning course project, the solution
was evolved into a production-ready application by integrating **MLflow** for
experiment tracking, **FastAPI** for model serving, and **Docker** for containerization.

A Support Vector Machine (**SVM**) acts as the primary (**"master"**) classifier,
while an **XGBoost** model is selectively activated as an **assistant** only in
low-confidence regions of the SVM decision space. The assistant is designed
to correct systematic errors in a known hard classification region
(classes 2 vs 5), without degrading overall model stability.

*The project emphasizes on robustness and
reproducible evaluation, rather than purely optimizing peak accuracy.*


## Table of Contents
- [Overview](#overview)
- [Data Understanding](#data-understanding)
- [Model Architecture](#model-architecture)
- [Repository Structure](#repository-structure)
- [Training Pipeline](#training-pipeline)
- [Evaluation Strategy](#evaluation-strategy)
- [Stress Testing](#stress-testing)
- [Experiment Tracking](#experiment-tracking)
- [Inference & API Design](#inference--api-design)
- [Dockerized Deployment](#dockerized-deployment)
- [How to Run](#how-to-run)
- [License](#license)


## Overview

The task is a multi-class classification problem with five discrete classes
(1–5). The primary objective is to achieve high classification accuracy while
maintaining stable and predictable model behavior across validation splits.


## Data Understanding

Before modeling, the dataset was carefully inspected to ensure that:
- No missing values or corrupted samples were present
- Class distribution was balanced across all five classes
- Feature vectors were consistent in dimensionality (224 features per sample)

Dimensionality reduction techniques (e.g. t-SNE) and controlled experiments
with baseline models confirmed that the main source of misclassification
originates from the overlap between classes 2 and 5, while the remaining
classes are comparatively easy to separate.

Based on these observations, the classification problem was conceptually
decomposed into:
- an **"easy"** region (classes 1, 3, 4), where standard classifiers achieve
  consistently high accuracy
- a **"hard"** region (classes 2 vs 5), where decision boundaries are less stable
  and model uncertainty is significantly higher

*This distinction motivated the use of a confidence-aware ensemble strategy,
rather than a single model optimized for global accuracy.*


## Model Architecture

The system is built around an ensemble architecture with explicit
role separation between models.

A Support Vector Machine (SVM) serves as the master classifier and is
responsible for producing predictions for all samples. An XGBoost model is
introduced as an assistant, designed to intervene only when the SVM exhibits
low confidence in its decision.

### SVM as Master
The SVM was selected as the master model due to its strong and stable
performance across the full dataset, particularly in the easy classification
region (classes 1, 3, 4).

### Gating Mechaninsm 
For samples predicted as either class 2 or 5, model confidence is estimated
using the absolute difference between the predicted class probabilities for
these two classes.

If this margin exceeds a predefined threshold **τ**, the SVM prediction is
accepted without modification. Otherwise, the sample is considered
low-confidence and is eligible for correction by the assistant model.

### XGBoost as Assistant
The assistant model is an XGBoost classifier trained specifically to handle
the ambiguous region between classes 2 and 5.

The assistant is consulted only for low-confidence samples,
limiting its influence to cases where the master
model is most likely to fail.

### Weighted Voting
When the assistant is activated, the final prediction is determined through
a weighted combination of the SVM and XGBoost outputs (probabilities), controlled by a
parameter w.

*This design allows fine-grained control over the degree of correction,
preventing over-reliance on the assistant and reducing the risk of introducing
noise into high-confidence predictions (Note: XGBoost tends to be overconfident).*


## Repository Structure

The repository is organized as follows to separate training and logging,
evaluation, and inference concerns:
```
ensemble-classifier/
│
├── config/
│   └── config.yaml              # Centralized configuration for training & evaluation
│
├── data/
│   ├── datasetTV.csv             # Training dataset
│   └── datasetTest.csv           # Test dataset
│
├── src/
│   ├── data.py                   # Data loading and splitting
│   ├── model.py                  # Ensemble model implementation
│   ├── evaluation.py             # Evaluation & stress testing
│   ├── utils.py                  # Utilities (configuration processing)
│   └── __init__.py
│
├── training/
│   ├── train_and_log.py          # Training, evaluation & MLflow logging pipeline
│   └── __init__.py
│
├── inference/
│   ├── api.py                    # FastAPI inference service
│   ├── schemas.py                # Request/response schemas
│   ├── logging.py                # Inference-time logging
│   └── __init__.py
│
├── models/
│   ├── ensemble_model.pkl        # Trained ensemble model
│   ├── svm_model.pkl             # Trained SVM master model
│   └── xgb_model.pkl             # Trained XGBoost assistant model
│
├── artifacts/                    # Generated artifacts
│
├── Dockerfile                    # Inference-only Docker image
├── requirements.txt              # Project dependencies
├── requirements-inference.txt    # Minimal inference dependencies
└── README.md
```

## Training Pipeline

Model training follows a reproducible pipeline consisting of:
- data loading and train/validation split
- dimensionality reduction using PCA
- training of the SVM master model
- training of the XGBoost assistant on the ambiguous region (classes 2 vs 5)

*All training steps are parameterized via a configuration file to ensure
reproducibility and controlled experimentation.*

## Evaluation Strategy

Model evaluation goes beyond global accuracy and explicitly measures behavior
in both easy and hard classification regions.

The following metrics are reported:
- Global accuracy across all classes
- Hard accuracy (classes 2 vs 5)
- Easy accuracy (classes 1, 3, 4)
- Assistant activation rate
- Assistant correction rate


## Stress Testing

To assess robustness and variance, the system is evaluated using a repeated
stratified k-fold stress test.

For each fold, a fresh ensemble is trained and evaluated, and the mean and
standard deviation of accuracy are reported. This provides a more reliable
estimate of real-world performance compared to a single validation split.

*Stress test results consistently show high mean accuracy with low variance,
indicating stable model behavior under data perturbations.*


## Experiment Tracking

All experiments are tracked using MLflow to ensure reproducibility and
systematic comparison between runs.

Each training run logs:
- model hyperparameters
- ensemble configuration (τ, w)
- validation metrics (global, hard, easy accuracy)
- assistant behavior metrics
- stress test statistics (mean and standard deviation)

In addition to scalar metrics, the following artifacts are logged:
- confusion matrices
- stress test summaries
- evaluation diagnostics

The training pipeline is fully configuration-driven and deterministic, allowing
experiments to be reproduced by re-running the training script with the same
configuration file.

**MLflow integration enables structured experimentation and prevents ad-hoc
model selection, making the development process production-
oriented rather than exploratory-only.*


## Inference & API Design

The trained ensemble is exposed through a lightweight inference API built
with FastAPI.

Inference logic is completelly separated from the training pipeline and
operates exclusively on pre-trained models

The API exposes a **single prediction** endpoint that accepts **fixed-length**
feature vectors (224 features per sample) and returns:
- the final predicted class
- the SVM confidence margin (Warning: margin only between p2 and p5 even when predicted class is different)
- an indicator of assistant model activation

Example API response:

```json
{
  "prediction": 5,
  "svm_margin": 0.32,
  "assistant_used": true
}
```

*The inference stack includes only the minimal dependencies required for
serving predictions, ensuring a lightweight and maintainable deployment
artifact.*

## Dockerized Deployment

To ensure portability and environment consistency, the inference service is
fully containerized using Docker.

The Docker image is intentionally designed for inference only and includes
a minimal set of dependencies required to serve predictions via the API.

Training-related dependencies and tooling are excluded to keep the image
lightweight and focused.


## How to Run

The project is developed in Python 3.10+ and assumes a Windows or Unix-like
environment with virtual environment support.

### Environment Setup

Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate      # Linux / macOS

pip install -r requirements.txt
```

### Training & Experiment Tracking

To train the model, evaluate it, and log all results to MLflow:
```bash
python -m training.train_and_log
```
This will:
- train the ensemble model
- evaluate performance on a validation split
- perform a stress test using repeated stratified k-fold cross-validation
- log all metrics, parameters, and artifacts to MLflow

To inspect experiment results:
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

### Run Inference API

The trained model can be served locally using FastAPI:
```bash
uvicorn inference.api:app --reload
```
The API will be available at http://127.0.0.1:8000.

### Docker Inference

To build the inference-only Docker image:
```bash
docker build -t confidence-gated-ensemble .
```
To run the container:
```bash
docker run -p 8000:8000 confidence-gated-ensemble
```
This runs the FastAPI inference service in a containerized environment using
pre-trained models.


## License
