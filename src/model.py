import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class EnsembleModel:
    """
    Ensemble model for multi-class classification:
    - Global SVM with PCA for general predictions
    - Specialist XGBoost for 2 vs 5 sub-problem
    """

    def __init__(self, pca_components=27, svm_c=2, svm_kernel="rbf", svm_gamma='auto',
                 w_svm=0.8, tau=0.45,
                 xgb_n_estimators=800, xgb_max_depth=5, xgb_learning_rate=0.1,
                 xgb_subsample=0.8, xgb_colsample_bytree=0.8,
                 random_seed=42,
                 svm_model=None, xgb_model=None):

        self.pca_components = pca_components
        self.svm_c = svm_c
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma

        self.w_svm = w_svm
        self.tau = tau

        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree

        self.random_seed = random_seed

        self.svm_model = svm_model
        self.xgb_model = xgb_model

    def train_global_svm(self, X_train, y_train):
        """
        Train the global SVM model using StandardScaler + PCA pipeline.
        """
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=self.pca_components)),
            ("svm", SVC(
                C=self.svm_c,
                kernel=self.svm_kernel,
                gamma=self.svm_gamma,
                probability=True,
                random_state=self.random_seed
            ))
        ])
        pipeline.fit(X_train, y_train)
        self.svm_model = pipeline

    def train_specialist_xgb(self, X_train, y_train):
        """
        Train specialist XGBoost for 2 vs 5 sub-problem.
        """
        mask = y_train.isin([2, 5])
        X_spec = X_train[mask]
        y_spec = y_train[mask].map({2: 0, 5: 1})

        model = XGBClassifier(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            eval_metric="logloss",
            random_state=self.random_seed
        )
        model.fit(X_spec, y_spec)
        self.xgb_model = model

    def fit(self, X_train, y_train):
        """
        Train both the global SVM and specialist XGBoost models.
        """
        self.train_global_svm(X_train, y_train)
        self.train_specialist_xgb(X_train, y_train)

    def predict(self, X):
        """
        Predict labels using ensemble logic:
        - SVM makes primary prediction
        - For classes 2 and 5 with low confidence, XGB is consulted
        """
        if self.svm_model is None or self.xgb_model is None:
            raise ValueError("Models not trained. Fit first.")

        svm_probs = self.svm_model.predict_proba(X)
        svm_preds = self.svm_model.predict(X)
        final_preds = svm_preds.copy()

        cls_idx = {c: i for i, c in enumerate(self.svm_model.classes_)}
        idx_2, idx_5 = cls_idx[2], cls_idx[5]

        assistant_activation_count = 0
        assistant_prediction_change = 0
        assistant_activation_rate = 0
        assistant_change_rate = 0
        count = 0
        for i in range(len(X)):
            if svm_preds[i] not in (2, 5):
                continue
            p2_svm = svm_probs[i, idx_2]
            p5_svm = svm_probs[i, idx_5]
            margin = abs(p2_svm - p5_svm)
            count += 1

            if margin >= self.tau:
                continue
            assistant_activation_count += 1
            p2_xgb, p5_xgb = self.xgb_model.predict_proba(X.iloc[[i]])[0]

            p2 = self.w_svm * p2_svm + (1 - self.w_svm) * p2_xgb
            p5 = self.w_svm * p5_svm + (1 - self.w_svm) * p5_xgb

            final_preds[i] = 2 if p2 >= p5 else 5

            if final_preds[i] != svm_preds[i]:
                assistant_prediction_change += 1

            assistant_activation_rate = assistant_activation_count / count
            assistant_change_rate = assistant_prediction_change / assistant_activation_count
        return final_preds, assistant_activation_rate, assistant_change_rate

    def save_models(self, path_svm="svm_model.pkl", path_xgb="xgb_model.pkl"):
        """
        Save trained models.
        """
        if self.svm_model is not None:
            joblib.dump(self.svm_model, path_svm)
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, path_xgb)

    def load_models(self, path_svm="svm_model.pkl", path_xgb="xgb_model.pkl"):
        """
        Load trained models.
        """
        self.svm_model = joblib.load(path_svm)
        self.xgb_model = joblib.load(path_xgb)

    def evaluate_inference(self, X):
        """
        Single-sample inference.
        Returns:
            prediction (int)
            svm_margin (float)
            assistant_used (bool)
        """

        svm_probs = self.svm_model.predict_proba(X)[0]

        cls_idx = {c: i for i, c in enumerate(self.svm_model.classes_)}
        p2 = svm_probs[cls_idx[2]]
        p5 = svm_probs[cls_idx[5]]

        # margin of 2 and 5 probabilities
        svm_margin = abs(p2 - p5)
        prediction = self.svm_model.predict(X)[0]

        assistant_used = False

        if prediction in (2, 5) and svm_margin < self.tau:
            assistant_used = True

            xgb_probs = self.xgb_model.predict_proba(X)[0]
            p2_xgb, p5_xgb = xgb_probs

            p2_final = self.w_svm * p2 + (1 - self.w_svm) * p2_xgb
            p5_final = self.w_svm * p5 + (1 - self.w_svm) * p5_xgb

            prediction = 2 if p2_final >= p5_final else 5

        return prediction, svm_margin, assistant_used


