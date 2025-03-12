import requests
import json
from typing import Dict, Any, Tuple


def predict_income(
    data: Dict[str, Any],
    url: str = "https://scalable-ml-pipe-census.onrender.com//predict",
) -> Tuple[Dict[str, Any], int]:
    """
    Send a POST request to the Census Income Prediction API.

    Args:
        data: Dictionary containing the census data
        url: API endpoint URL

    Returns:
        Tuple containing the response JSON and status code
    """
    try:
        response = requests.post(url, json=data)
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}, 500


def main():
    # Example census data
    sample_data = {
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
        "native-country": "United-States",
    }

    # Make prediction
    result, status_code = predict_income(sample_data)

    # Print results
    print(f"Status Code: {status_code}")
    print("Prediction Results:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
