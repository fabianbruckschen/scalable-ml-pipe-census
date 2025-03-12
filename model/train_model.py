# Script to train machine learning model.
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from ml.data import process_data
from ml.model import train_model, inference, compute_and_log_metrics_by_slice

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("slice_output.txt"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    try:
        # load in the data.
        logger.info("Loading Data...")
        data = pd.read_csv("../data/census.csv")
        data = data.replace("?", np.nan)
        data = data.fillna(data.mode().iloc[0])

        # split and transform
        logger.info("Splitting and transforming data...")
        train, test = train_test_split(data, test_size=0.20, random_state=42)

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )

        X_test, y_test, _, _ = process_data(
            test,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # train
        logger.info("Training model...")
        model = train_model(X_train, y_train)

        # evaluate
        logger.info("Evaluating model...")
        y_pred = inference(model, X_test)
        logger.info(
            f"\nClassification Report:\n{classification_report(y_test, y_pred)}"
        )
        compute_and_log_metrics_by_slice(
            model, X_test, y_test, list(data.columns), cat_features, logger
        )

        # Save artifacts
        logger.info("Saving model artifacts...")

        joblib.dump(model, "artifacts/model.joblib")
        joblib.dump(encoder, "artifacts/encoder.joblib")
        joblib.dump(lb, "artifacts/lb.joblib")
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
