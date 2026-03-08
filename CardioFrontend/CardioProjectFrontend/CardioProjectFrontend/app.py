from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load scaler and model
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        id_val = 0 # Default random ID since it doesn't matter for prediction
        # The frontend will calculate age in days (since the dataset age is in days)
        age = float(request.form.get('age_days')) 
        gender = float(request.form.get('gender'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        ap_hi = float(request.form.get('ap_hi'))
        ap_lo = float(request.form.get('ap_lo'))
        cholesterol = float(request.form.get('cholesterol'))
        gluc = float(request.form.get('gluc'))
        smoke = float(request.form.get('smoke'))
        alco = float(request.form.get('alco'))
        active = float(request.form.get('active'))
        
        # Order must exactly match scaler's expected features:
        # ['id' 'age' 'gender' 'height' 'weight' 'ap_hi' 'ap_lo' 'cholesterol' 'gluc' 'smoke' 'alco' 'active']
        input_features = [id_val, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]
        
        # Convert to numpy array
        final_input = np.array([input_features])
        
        # Apply scaling
        final_input = scaler.transform(final_input)
        
        # Make prediction
        prediction = model.predict(final_input)

        if prediction[0] == 1:
            result = "High Risk of Cardiovascular Disease"
        else:
            result = "Low Risk of Cardiovascular Disease"
            
    except Exception as e:
        result = f"Error processing input: {str(e)}"
    return render_template("result.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)