from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API"}


def test_predict_below_50k():
    """Test prediction endpoint with data that should predict income <= 50K."""
    data = {
        "age": 39,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "HS",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_predict_above_50k():
    """Test prediction endpoint with data that should predict income >50K."""
    data = {
        "age": 42,
        "workclass": "Private",
        "fnlgt": 159449,
        "education": "BSc",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5178,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_predict_invalid_data():
    """Test prediction endpoint with invalid data."""
    data = {
        "age": "invalid",  # age should be int
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_predict_missing_field():
    """Test prediction endpoint with missing required field."""
    data = {
        "age": 39,
        "workclass": "Private",
        "fnlgt": 77516,
        # education field missing
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422
