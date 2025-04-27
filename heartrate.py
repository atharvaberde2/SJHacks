from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import google.generativeai as genai
from flask_cors import CORS
import os
import pandas as pd

model1 = joblib.load("C:/Users/athar/Downloads/heart_risk_model.pkl")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = os.urandom(24)

genai.configure(api_key= "AIzaSyD6kMAT2p-u6IkT9aIusodwV9y9F_6jlMk")
prompt = ('''You are a virtual heart doctor specializing in cardiology.

When a user messages you with any symptoms, stats, or concerns (such as heart rate, chest pain, dizziness, fatigue, breathing difficulty), you should:

Interpret their input carefully.

Ask smart follow-up questions to get more detailed information (ex: "How long have you felt this?", "Are you experiencing shortness of breath?", etc).

Estimate a possible diagnosis based on the information.

Suggest safe next steps, such as resting, seeking urgent care, or seeing a doctor.

Always recommend consulting a real doctor for serious concerns.

Use a warm, reassuring tone — sound like a friendly and knowledgeable heart specialist.

Respond in short, clear paragraphs — easy for the user to understand quickly. ''')

model = genai.GenerativeModel("gemini-1.5-pro")  # Keep model separate

chat_session = model.start_chat(history=[
    {"role": "user", "parts": [prompt]}
])


@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Assume model is already loaded
        input_data = [
            data['age'], 
            data['gender'], 
            data['impulse'], 
            data['pressureHigh'], 
            data['pressureLow'], 
            data['glucose'], 
            data['kcm'], 
            data['troponin']
        ]

        input_df = pd.DataFrame([input_data], columns=[
            'age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin'
        ])

        prediction = model1.predict(input_df)[0]
        prediction = int(prediction)
        if prediction == 1:
            prediction = "High Risk"
        elif prediction == 0:
            prediction = "Low Risk"

        proba = model1.predict_proba(input_df)
        confidence = float(proba.max()) *100

        
        return jsonify({
            'predicted_heart_rate': prediction,
             'confidence': confidence
        })
    except Exception as e:
        print("Error occurred:", e)   # 👈 prints the real Python error
        return jsonify({'error': str(e)}), 500

@app.route('/chat (1).html', methods=['GET'])
def chat():
    return render_template('chat (1).html')


@app.route('/chatH', methods=['POST'])
def chatH():
    # Get the JSON data from the request
    data = request.json
    user_message = data.get("message", "")

    # Check if the message is empty
    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        response = chat_session.send_message(user_message)
        return jsonify({"response": response.text})

    except Exception as e:
        # Log the error for debugging purposes
        print("Error occurred:", e) 
        return jsonify({"error": str(e)}), 500


@app.route('/call.html', methods=['GET'])
def call():
    return render_template('call.html')



if __name__ == "__main__":
    app.run(debug=True)