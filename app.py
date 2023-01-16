import streamlit as st
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


st.title("Gender Prediction Based on Names: ")

def preprocess(names_df, train=True):
    # Step 1: Lowercase
    names_df['Name'] = names_df['Name'].str.lower()

    # Step 2: Split individual characters
    names_df['Name'] = [list(name) for name in names_df['Name']]

    # Step 3: Pad names with spaces to make all names same length
    name_length = 50
    names_df['Name'] = [
        (name + [' ']*name_length)[:name_length]
        for name in names_df['Name']
    ]

    # Step 4: Encode Characters to Numbers
    names_df['Name'] = [
        [
            max(0.0, ord(char)-96.0)
            for char in name
        ]
        for name in names_df['Name']
    ]

    if train:
        # Step 5: Encode Gender to Numbers
        names_df['Gender'] = [
            0.0 if gender == 'F' else 1.0
            for gender in names_df['Gender']
        ]

    return names_df


#Load the model
pred_model = load_model('boyorgirl_5.h5')

# Input names
names = st.text_input("Enter The Name")
names = names.split(",")
if st.button("Get Gender"):
    # Convert to dataframe
    pred_df = pd.DataFrame({'Name': names})

    # Preprocess
    pred_df = preprocess(pred_df, train=False)

    # Predictions
    result = pred_model.predict(np.asarray(
        pred_df['Name'].values.tolist())).squeeze(axis=1)

    pred_df['Predicted Gender'] = [
        'Boy' if logit > 0.5 else 'Girl' for logit in result
    ]

    pred_df['Prediction Confidence (0-1)'] = [
        logit if logit > 0.5 else 1.0 - logit for logit in result
    ]

    # Format the output
    pred_df['Name'] = names
    pred_df['Prediction Confidence (0-1)'] = pred_df['Prediction Confidence (0-1)'].round(2)
    pred_df.drop_duplicates(inplace=True)
    st.header('Predictions: ')
    st.table(pred_df)
