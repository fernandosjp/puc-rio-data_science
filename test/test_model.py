import pytest
from utils.transaction_classifier import classify_transaction, classify_transaction_list
from sklearn.metrics import accuracy_score
import pandas as pd 

transaction_categories = {
    "Gasoline": "Auto",
    "Mall": "Food",
    "McDonalds": "Food",
    "Water bottle": "Food",
    "Cheese burger": "Food",
    "Cellphone game": "Entertainment"
}

@pytest.mark.parametrize("description", transaction_categories.keys())
def test_basic_predictions(description):
    prediction = classify_transaction(description).get("Category", "")
    assert prediction == transaction_categories.get(description, "")


# test against model performance metric and a bigger test csv
def test_mode_accuracy():
    # Dataset URL
    url = "https://raw.githubusercontent.com/fernandosjp/puc-rio-data_science/main/test/test_data.csv"
    data = pd.read_csv('test/test_data.csv', delimiter=',', encoding='latin-1')
    data['Description'] = data['Description'].values.astype('U')

    features = ['Description'] 
    target = ['Category'] 

    X = data[features]
    y = data[target]
    yhat = classify_transaction_list(X.Description.values.tolist())
    accuracy = accuracy_score(yhat.Category.values.tolist(), y.Category.values.tolist())
    assert accuracy >= 0.85