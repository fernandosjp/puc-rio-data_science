from utils.transaction_classifier import classify_transaction


def test_model():
    description = "Habibs"
    assert classify_transaction(description) == "Restaurants"