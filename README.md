# Confidence-Gated Ensemble Classifier

This repository presents a production-oriented machine learning system
built around a confidence-gated ensemble architecture.

Originally designed as a Machine Learning course project, the solution
was evolved into a production-ready application by integrating MLflow for
experiment tracking, FastAPI for model serving, and Docker for containerization.

A Support Vector Machine (SVM) acts as the primary ("master") classifier,
while an XGBoost model is selectively activated as an assistant only in
low-confidence regions of the SVM decision space. The assistant is designed
to correct systematic errors in a known hard classification region
(classes 2 vs 5), without degrading overall model stability.

The project emphasizes on robustness and
reproducible evaluation, rather than purely optimizing peak accuracy.

