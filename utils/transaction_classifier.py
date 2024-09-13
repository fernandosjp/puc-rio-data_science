import joblib

model = joblib.load(open('model/production/model.joblib', 'rb'))
vectorizer = joblib.load(open('model/production/vectorizer.joblib', 'rb'))
label_encoder = joblib.load(open('model/production/label_encoder.joblib', 'rb'))

def classify_transaction(description):
    text_vec = vectorizer.transform([description])
    prediction = model.predict(text_vec)
    probability_score = model.predict_proba(text_vec)[0].max()
    decoded_label = label_encoder.inverse_transform(prediction)[0]
    return {"transaction": description, "label": decoded_label,"score": probability_score}