import pytest
from utils.transaction_classifier import classify_transaction

transaction_categories = {
    "Gas": "Utilities",
    "Walmart": "Food",
    "Market": "Food",
    "McDonald": "Food"
}

@pytest.mark.parametrize("description", transaction_categories.keys())
def test_basic_predictions(description):
    prediction = classify_transaction(description).get("label", "")
    assert prediction == transaction_categories.get(description, "")


# test against model performance metric and a bigger test csv