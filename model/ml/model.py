from logging import Logger

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.array, y_train: np.array) -> XGBClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Train XGBoost model
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective="binary:logistic",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y: np.array, preds: np.array) -> (float, float, float):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_and_log_metrics_by_slice(
    model: XGBClassifier,
    X: np.array,
    y: np.array,
    feature_names: list,
    cat_features: list,
    logger: Logger,
    min_samples: int = 30,
) -> None:
    """
    Computes and logs model metrics for slices of data based on categorical features.

    Inputs
    ------
    model : XGBClassifier
        Trained model
    X : np.array
        Features data
    y : np.array
        Labels
    feature_names : list
        List of all feature names
    cat_features : list
        List of categorical feature names
    logger : Logger
        Logger object for stdout
    min_samples : int
        Minimum number of samples required to compute metrics for a slice
    """
    for feature in cat_features:
        logger.info(f"\nMetrics for feature: {feature}")
        logger.info("=" * 50)

        feature_idx = feature_names.index(feature)
        unique_values = np.unique(X[:, feature_idx])

        for unique_value in unique_values:
            mask = X[:, feature_idx] == unique_value

            if mask.sum() > min_samples:
                X_slice = X[mask]
                y_slice = y[mask]
                preds_slice = inference(model, X_slice)

                precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)

                logger.info(f"\nSlice: {unique_value}")
                logger.info(f"Number of samples: {len(X_slice)}")
                logger.info(f"Precision: {precision:.3f}")
                logger.info(f"Recall: {recall:.3f}")
                logger.info(f"F-beta: {fbeta:.3f}")
                logger.info("-" * 30)


def inference(model: XGBClassifier, X: np.array) -> np.array:
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : XGBClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
