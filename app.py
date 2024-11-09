from flask import Flask, request, jsonify
import numpy as np
from joblib import dump, load

# Load your pre-trained model
model = load('test_model.joblib')

# Initialize Flask app
app = Flask(__name__)


# Endpoint for predicting sales
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the POST request
    data = request.json

    # Extract features for prediction (TV, Radio, Newspaper)
    features = np.array([data['TV'], data['Radio'], data['Newspaper']])

    # Make prediction
    prediction = model.predict([features])

    # Return prediction result as JSON
    return jsonify({'prediction': prediction[0]})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
