import pytest
from utils.transaction_classifier import classify_transaction

transaction_categories = {
    "Habibs": "Restaurants",
    "Gasolina": "Transporte",
    "Estacionamento": "Transporte",
    "Lanche": "Comida",
    "Carrefour": "Mercado",
    "Shopper": "Mercado"
}

@pytest.mark.parametrize("description", transaction_categories.keys())
def test_basic_predictions(description):
    assert classify_transaction(description) == transaction_categories.get(description, "")


# test against model performance metric and a bigger test csv