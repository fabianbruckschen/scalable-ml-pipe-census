# Census Income Prediction API

This repository contains a machine learning API that predicts whether an individual's income exceeds $50K/year based on census data. The project implements a scalable ML pipeline using FastAPI, XGBoost, scikit-learn, and Python.

## Project Structure
```
scalable-ml-pipe-census/
├── main.py                # FastAPI application
├── test_main.py           # API endpoint tests
├── model/
│   ├── artifacts/         # Saved model files
│   │   ├── model.joblib
│   │   ├── encoder.joblib
│   │   └── lb.joblib
│   └── ml/
│       ├── data.py        # Data processing utilities
│       └── model.py       # Model inference logic
```
## Features

- RESTful API built with FastAPI
- Input validation using Pydantic models
- Automated testing with pytest
- Model inference endpoint
- Proper error handling and type hints
- Categorical variable encoding

## Installation

1. Clone the repository:
```bash
git clone git@github.com:fabianbruckschen/scalable-ml-pipe-census.git
cd scalable-ml-pipe-census
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -f requirements.txt
```
4. Activation
```bash
uvicorn main:app --reload
```
5. Example request body:
```json
{
    "age": 35,
    "workclass": "Private",
    "fnlgt": 122273,
    "education": "BSc",
    "education-num": 13,
    "marital-status": "Married",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
```
6. Making Predictions (You can use the provided post_to_api.py script to make predictions:)
```bash
python live_post.py
```
7. Testing
```bash
pytest test_main.py -v
```

## Model Details
The model is pre-trained and saved using joblib. It includes:

### Main model classifier
- Feature encoder for categorical variables
- Label binarizer for the target variable

## Error Handling
The API implements proper error handling for:

- Invalid input data
- Missing required fields
- Model inference errors
- Server errors
- Contributing
- Fork the repository
- Create a feature branch
- Commit your changes
- Push to the branch
- Create a Pull Request
