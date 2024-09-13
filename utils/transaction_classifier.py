import joblib
import numpy as np
import pandas as pd

model_version = 'v2'

model = joblib.load(open(f'model/production/{model_version}-model.joblib', 'rb'))
vectorizer = joblib.load(open(f'model/production/{model_version}-vectorizer.joblib', 'rb'))
label_encoder = joblib.load(open(f'model/production/{model_version}-label_encoder.joblib', 'rb'))

def classify_transaction(description):
    text_vec = vectorizer.transform([description])
    prediction = model.predict(text_vec)
    probability_score = model.predict_proba(text_vec)[0].max()
    decoded_label = label_encoder.inverse_transform(prediction)[0]
    return {"Description": description, "Category": decoded_label,"Score": probability_score}

def classify_transaction_list(description_list):
    """"

    Returns a dataframe with results
    """
    list_for_model_output = []

    for row in description_list:
        model_output = classify_transaction(row)
        list_for_model_output.append(model_output)
    
    df = pd.DataFrame.from_dict(list_for_model_output)

    # # List comprehension to convert the score from decimals to percentages
    f = [f"{row:.2%}" for row in df["Score"]]
    df["Accuracy"] = f
    df.drop(["Score"], inplace=True, axis=1)

    # # We need to change the index. Index starts at 0, so we make it start at 1
    df.index = np.arange(1, len(df) + 1)

    return df