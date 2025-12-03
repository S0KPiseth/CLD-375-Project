from flask import Flask, jsonify, request
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import traceback 

path_to_model = "Gradient-Boosting-diabetes-Classifier.pkl"
path_to_scaler = "scaler_filename.pkl"

le = LabelEncoder()

try:
    with open(path_to_scaler, 'rb') as file:
        loaded_scaler = pickle.load(file)

    with open(path_to_model, 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error loading pickle files: {e}")

def data_preprocessing(df):
    data = df.copy()
    
    cols_to_drop = ['bmi', 'smoking_history', "gender"]
    if all(col in data.columns for col in cols_to_drop):
        data = data.drop(cols_to_drop, axis=1)
    else:
        print("Warning: Some columns to drop were not found.")

    non_categorical = ["age", "HbA1c_level", "blood_glucose_level"]
    
    try:
        data[non_categorical] = loaded_scaler.transform(data[non_categorical])
    except Exception as e:
        print(f"Error during scaling transform: {e}")
        
    return data

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data:
        expected_keys = [
            'gender', 'age', 'hypertension', 'heart_disease',
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
        ]
        if not all(key in data for key in expected_keys):
            return jsonify({"error": "Input JSON is missing required fields."}), 400
        df = pd.DataFrame.from_dict(data, orient='index').T
        
        try:
            input_data_scaled = data_preprocessing(df)
            
            prediction = loaded_model.predict(input_data_scaled)
            print(f"Prediction result: {prediction}")
            
            return jsonify({"prediction": prediction.tolist()}), 200

        except Exception as e:
            traceback.print_exc() 
            return jsonify({"error": f"Prediction failed due to internal error: {str(e)}"}), 500
    
    return jsonify({"error": "No JSON data received"}), 400

if __name__ == '__main__':
    app.run(debug=True)

