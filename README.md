# Financial transactions classifier

Web app with machine learning model to classify financial transactions into categories. This Machine Learning model will be deployed using Streamlit framework to allow simple and practical way of exposing the Machine Learning model to a web app. 

## Usage
Use virtualenv to install requirements and run Streamlit application to test the classification model throught a front-end. 

```bash
streamlit run streamlit_app.py
```

## Model Training
Model was trained using a Google Colab available in the following link: [Transaction Classifier Notebook](https://colab.research.google.com/github/fernandosjp/puc-rio-data_science/blob/main/model/model_training_notebook.ipynb)

## Test
After updating joblib files in `model/production` folder, run the test to assure its quality. The test will check the categorization of a samll sample of common transactions and alse calculate the accuracy against an unseen test dataset to make sure metric is above 85% which is the threshold defined for minimum model quality.

```bash
pytest
```

## References

Dataset from:
https://github.com/j-convey/BankTextCategorizer/tree/main

For web application:
https://github.com/streamlit/example-app-zero-shot-text-classifier
