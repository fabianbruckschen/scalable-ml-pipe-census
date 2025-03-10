# test_train_model.py
import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from .ml.data import process_data
from .ml.model import train_model, inference, compute_model_metrics


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "salary": np.random.choice(
                ["<=50K", ">50K"], size=n_samples, p=[0.6, 0.4]
            ),  # 60-40 split
            "age": np.random.randint(18, 70, n_samples),
            "workclass": np.random.choice(
                ["Private", "Public", "Self-employed"],
                size=n_samples,
                p=[0.6, 0.3, 0.1],
            ),
            "education": np.random.choice(
                ["HS", "BSc", "MSc", "PhD"], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]
            ),
            "race": np.random.choice(
                ["White", "Black", "Asian", "Other"],
                size=n_samples,
                p=[0.5, 0.2, 0.2, 0.1],
            ),
            "sex": np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5]),
        }
    )

    # Make salary more realistic based on education and age
    conditions = [
        (data["education"].isin(["MSc", "PhD"]) & (data["age"] > 35)),
        (data["education"].isin(["BSc"]) & (data["age"] > 30)),
        (data["education"].isin(["HS"]) & (data["age"] > 45)),
    ]
    data.loc[conditions[0] | conditions[1] | conditions[2], "salary"] = ">50K"
    return data


@pytest.fixture
def processed_data(sample_data):
    """Create processed data for testing"""
    cat_features = ["workclass", "education", "race", "sex"]
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train, encoder, lb


def test_process_data(sample_data):
    """Test data processing function"""
    cat_features = ["workclass", "education", "race", "sex"]
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )

    assert X_train.shape[0] == 100  # number of samples
    assert y_train.shape[0] == 100
    assert encoder is not None
    assert lb is not None


def test_train_model(processed_data):
    """Test model training function"""
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)

    assert isinstance(model, XGBClassifier)
    assert hasattr(model, "predict")


def test_inference(processed_data):
    """Test model inference function"""
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    predictions = inference(model, X_train)

    assert len(predictions) == X_train.shape[0]
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)


def test_compute_model_metrics():
    """Test metrics computation"""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_model_performance(processed_data):
    """Test overall model performance"""
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    predictions = inference(model, X_train)
    print(y_train, predictions)
    precision, recall, fbeta = compute_model_metrics(y_train, predictions)

    # Basic sanity checks for model performance
    assert precision > 0
    assert recall > 0
    assert fbeta > 0
