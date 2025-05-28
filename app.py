from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import warnings

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load your trained model and scaler
try:
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    print("Make sure both 'svm_model.pkl' and 'scaler.pkl' exist in the same directory.")
    model = None
    scaler = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "SVM Model API is running!",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/ (GET)"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            return jsonify({
                "error": "Model or scaler not loaded",
                "message": "Please ensure both model and scaler files exist"
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'features' not in data:
            return jsonify({
                "error": "Invalid input",
                "message": "Please provide 'features' in JSON format"
            }), 400
        
        # Extract features from the data
        features = data['features']
        
        # Validate input - breast cancer dataset has 30 features
        if len(features) != 30:
            return jsonify({
                "error": "Invalid feature count",
                "message": f"Expected 30 features for breast cancer dataset, got {len(features)}"
            }), 400
        
        # Convert to numpy array and reshape for single prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Apply scaling (suppress the feature names warning since it doesn't affect functionality)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            features_scaled = scaler.transform(features_array)
        
        # Make prediction on scaled features
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability
        try:
            prediction_proba = model.predict_proba(features_scaled)[0].tolist()
        except:
            prediction_proba = None
        
        # Convert prediction to human-readable format
        diagnosis = "Malignant" if prediction == 1 else "Benign"
        
        # Return prediction
        response = {
            "prediction": int(prediction),
            "diagnosis": diagnosis,
            "confidence": max(prediction_proba) * 100 if prediction_proba else None,
            "probabilities": {
                "benign": prediction_proba[0] if prediction_proba else None,
                "malignant": prediction_proba[1] if prediction_proba else None
            } if prediction_proba else None,
            "status": "success"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500
    
    try:
        info = {
            "model_type": str(type(model).__name__),
            "dataset": "Breast Cancer Wisconsin (Diagnostic)",
            "feature_count": 30,
            "classes": ["Benign", "Malignant"],
            "class_mapping": {"0": "Benign", "1": "Malignant"},
            "scaler_used": True,
            "scaler_type": str(type(scaler).__name__)
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Check if model files exist
    if not os.path.exists('svm_model.pkl'):
        print("Warning: svm_model.pkl not found!")
        print("Please save your trained model using: joblib.dump(your_model, 'svm_model.pkl')")
    
    if not os.path.exists('scaler.pkl'):
        print("Warning: scaler.pkl not found!")
        print("Please save your scaler using: joblib.dump(scaler, 'scaler.pkl')")
    
    app.run(debug=True, host='0.0.0.0', port=5000)