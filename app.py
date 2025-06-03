from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load('/home/divyansh/Desktop/ee/final_prototype/label_encoded_model.pkl')
input_encoders = joblib.load('/home/divyansh/Desktop/ee/final_prototype/input_label_encoders.pkl')
output_encoders = joblib.load('/home/divyansh/Desktop/ee/final_prototype/output_label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive form inputs (key names must match model training data)
    main_condition = request.form['Main_Condition']
    subtype = request.form['Subtype']

    input_data = {
        'Main_Condition': main_condition,
        'Subtype': subtype
    }
    
    print("Form input data:", input_data)

    # Prepare DataFrame
    df = pd.DataFrame([input_data])
    print("Before encoding:\n", df)

    # Encode inputs
    for col in input_encoders:
        try:
            df[col] = input_encoders[col].transform(df[col])
        except Exception as e:
            print(f"Encoding error in column '{col}': {e}")

    print("After encoding:\n", df)

    # Predict
    try:
        pred_encoded = model.predict(df)[0]
        print("Encoded prediction output:", pred_encoded)
    except Exception as e:
        print("Prediction error:", e)
        return render_template('index.html', prediction={"Error": str(e)}, inputs=input_data)

    # Decode predictions
    decoded = {}
    for i, col in enumerate(output_encoders):
        try:
            decoded[col] = output_encoders[col].inverse_transform([pred_encoded[i]])[0]
        except Exception as e:
            decoded[col] = f"Error decoding {col}: {e}"
            print(f"Decoding error in '{col}':", e)

    print("Decoded predictions:\n", decoded)

    return render_template('index.html', prediction=decoded, inputs=input_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
