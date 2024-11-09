import streamlit as st
import numpy as np
from joblib import load

# Load your pre-trained model (make sure 'test_model.joblib' is in the same directory)
model = load('test_model.joblib')

# Streamlit app code
def main():
    st.title("Sales Prediction App")
    st.write("Predict sales based on advertising budgets for TV, Radio, and Newspaper.")

    # Input fields for features
    tv_budget = st.number_input("TV Advertising Budget ($)", min_value=0.0, step=1.0)
    radio_budget = st.number_input("Radio Advertising Budget ($)", min_value=0.0, step=1.0)
    newspaper_budget = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, step=1.0)

    # Predict button
    if st.button("Predict"):
        # Prepare input data
        features = np.array([[tv_budget, radio_budget, newspaper_budget]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Display result
        st.success(f"Predicted Sales: ${prediction:.2f}")

if __name__ == '__main__':
    main()
