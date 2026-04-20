import streamlit as st
import pickle
import pandas as pd
import os

# --- Configuration --- #
MODEL_PATH = 'linear_regression_model(2).pkl'
ENCODERS_PATH = 'label_encoders.pkl'

# --- App Title --- #
st.title("Salary Prediction App")

# --- Load Model and Encoders --- #
model = None
label_encoders = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    st.success("Prediction model loaded successfully!")
else:
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure it's saved in the same directory.")
    st.stop()

if os.path.exists(ENCODERS_PATH):
    with open(ENCODERS_PATH, 'rb') as file:
        label_encoders = pickle.load(file)
    st.success("Label encoders loaded successfully!")
else:
    st.error(f"Label encoders file '{ENCODERS_PATH}' not found. Please ensure it's saved in the same directory.")
    st.stop()

# --- Sidebar for Input Features --- #
st.sidebar.header("Input Features")

# Numerical Inputs
rating = st.sidebar.slider("Rating", min_value=1.0, max_value=5.0, value=3.8, step=0.1)
salaries_reported = st.sidebar.number_input("Salaries Reported", min_value=1, value=3, step=1)

# Categorical Inputs - using loaded label encoders to get options
input_features = {}
categorical_cols = ['Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']

for col in categorical_cols:
    if col in label_encoders:
        # Get original categories to display in selectbox
        options = list(label_encoders[col].classes_)
        selected_option = st.sidebar.selectbox(f"Select {col}", options)
        input_features[col] = selected_option
    else:
        st.sidebar.text_input(f"Enter {col} (Encoder not found)", "") # Fallback if encoder somehow missing

# --- Prediction Button --- #
if st.sidebar.button("Predict Salary"):
    if model and label_encoders:
        # Prepare input data
        processed_inputs = {
            'Rating': rating,
            'Salaries Reported': salaries_reported
        }

        # Encode categorical inputs
        for col in categorical_cols:
            if col in label_encoders and col in input_features:
                try:
                    processed_inputs[col] = label_encoders[col].transform([input_features[col]])[0]
                except ValueError:
                    st.error(f"Selected '{input_features[col]}' for '{col}' is not a known category. Please select from the dropdown.")
                    st.stop()
            elif col in input_features: # If encoder not found but input was taken, this is a problem
                st.error(f"Cannot encode '{col}'. Label encoder for this column is missing.")
                st.stop()

        # Ensure the order of columns matches the training data
        # Based on X.columns: 'Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'
        final_input_df = pd.DataFrame([processed_inputs])
        final_input_df = final_input_df[['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']]

        # Make prediction
        prediction = model.predict(final_input_df)

        st.subheader("Predicted Salary")
        st.write(f"The predicted salary is: **₹{prediction[0]:,.2f}**")
    else:
        st.error("Model or label encoders not loaded. Cannot make prediction.")
