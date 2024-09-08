# Financial transactions classifier

Web app with machine learning model to classify financial transactions into categories. This Machine Learning model will be deployed using Streamlit framework to allow simple and practical way of exposing the Machine Learning model to a web app. 

## Usage
Run Streamlit application to test the classification model throught a front-end. 

```bash
streamlit run streamlit_app.py
```

## Test
After updating the model pickle in `model/model.pkl`, run the test to assure its quality. 

```bash
pytest
```

## References

For machine learning model:
https://github.com/j-convey/BankTextCategorizer/tree/main

For web application:
https://github.com/streamlit/example-app-zero-shot-text-classifier