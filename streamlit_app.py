import streamlit as st
import pandas as pd
import numpy as np
from utils.transaction_classifier import classify_transaction_list


############ PAGE LAYOUT AND TITLE ############

st.set_page_config(
    layout="centered", page_title="Splitfy transaction classifier", page_icon="❄️"
)

############ LOGO AND HEADING ############

st.image(
        "images/logo.png",
        width=200,
    )

st.title("Splitfy transaction classifier")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ SIDEBAR CONTENT ############

st.sidebar.write(
    """

App created by [fernandosjp](https://github.com/fernandosjp) using [Streamlit](https://streamlit.io/).

Inspired in example [Zero Shot Classifier](https://github.com/streamlit/example-app-zero-shot-text-classifier).

"""
)

############ TABBED NAVIGATION ############

MainTab, InfoTab = st.tabs(["Main", "Info"])

with InfoTab:

    st.subheader("What is Streamlit?")
    st.markdown(
        "[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications in Python."
    )

    st.subheader("Resources")
    st.markdown(
        """
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
    - [Book](https://www.amazon.com/dp/180056550X) (Getting Started with Streamlit for Data Science)
    """
    )

    st.subheader("Deploy")
    st.markdown(
        "You can quickly deploy Streamlit apps using [Streamlit Community Cloud](https://streamlit.io/cloud) in just a few clicks."
    )


with MainTab:

    st.write("")
    st.markdown(
        """

    Classify transactions on the fly to test the Splitfy Transactions Classifier.

    """
    )
    st.write("")

    with st.form(key="my_form"):

        # The block of code below is to display some text samples to classify.
        # This can of course be replaced with your own text samples.

        # MAX_KEY_PHRASES is a variable that controls the number of phrases that can be pasted:
        # The default in this app is 50 phrases. This can be changed to any number you like.

        MAX_DESCRIPTIONS = 50
        new_line = "\n"
        pre_defined_descriptions = [
            "Gas",
            "Market",
            "McDonald",
            "Walmart",
            "Water bottle",
            "Cheese burger",
            "Cellphone"
        ]
        sample_descriptions_string = f"{new_line.join(map(str, pre_defined_descriptions))}"

        # The block of code below displays a text area
        # So users can paste their phrases to classify

        text = st.text_area(
            # Instructions
            "Enter transactions descriptions to classify",
            sample_descriptions_string,
            height=200,
            help="At least one transaction for the classifier to work, one per line, "
            + str(MAX_DESCRIPTIONS)
            + " transaction descriptions max in 'unlocked mode'.",
            key="1",
        )

        # The block of code below:

        # 1. Converts the data st.text_area into a Python list.
        # 2. It also removes duplicates and empty lines.
        # 3. Raises an error if the user has entered more lines than in MAX_KEY_PHRASES.

        text = text.split("\n")  # Converts the pasted text to a Python list
        linesList = []  # Creates an empty list
        for x in text:
            linesList.append(x)  # Adds each line to the list
        linesList = list(dict.fromkeys(linesList))  # Removes dupes
        linesList = list(filter(None, linesList))  # Removes empty lines

        if len(linesList) > MAX_DESCRIPTIONS:
            st.info(
                f"❄️ Note that only the first "
                + str(MAX_DESCRIPTIONS)
                + " transaction descriptions will be reviewed to preserve performance."
            )

            linesList = linesList[:MAX_DESCRIPTIONS]

        submit_button = st.form_submit_button(label="Submit")

    ############ CONDITIONAL STATEMENTS ############
    # Conditional statements to check if users have entered valid inputs.
    # E.g. If the user has pressed the 'submit button without text, without labels, and with only one label etc.
    # The app will display a warning message.

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and len(text)==1 and text[0]=="":
        st.warning("❄️ There is no keyphrases to classify")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:

            # The block of code below if for our session state.
            # This is used to store the user's inputs so that they can be used later in the app.

            st.session_state.valid_inputs_received = True

        ############ CLASSIFY TRANSACTIONS USING ML MODEL ############

        df = classify_transaction_list(linesList)

        st.success("✅ Done!")

        st.caption("")
        st.markdown("### Check the results!")
        st.caption("")

        # Display the dataframe
        st.write(df)

        ############ DOWNLOAD BUTTON ############

        cs, c1 = st.columns([2, 2])
        with cs:

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(df)

            st.caption("")

            st.download_button(
                label="Download results in csv",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
            )
