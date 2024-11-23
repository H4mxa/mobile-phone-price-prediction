from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('phone_price_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

categorical_columns = ['No_of_sim', 'External_Memory', 'Android_version', 
                       'company', 'fast_charging', 'Processor_name']

def predict_price(input_data):
    input_data['Ram'] = float(input_data['Ram'])
    input_data['Battery'] = float(input_data['Battery'])
    input_data['Inbuilt_memory'] = float(input_data['Inbuilt_memory'])

    for col in categorical_columns:
        le = label_encoders[col]
        if input_data[col] not in le.classes_:
            input_data[col] = le.classes_[0] 
        input_data[col] = le.transform([input_data[col]])[0]

    input_data['Rating'] = input_data.get('Rating', 0)
    input_data['Spec_score'] = input_data.get('Spec_score', 0)

    input_features = pd.DataFrame([input_data], columns=model.feature_names_in_)

    predicted_price = model.predict(input_features)
    return predicted_price[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Ram': request.form['Ram'],
            'Battery': request.form['Battery'],
            'Inbuilt_memory': request.form['Inbuilt_memory'],
            'No_of_sim': request.form['No_of_sim'],
            'External_Memory': request.form['External_Memory'],
            'Android_version': request.form['Android_version'],
            'company': request.form['company'],
            'fast_charging': request.form['fast_charging'],
            'Processor_name': request.form['Processor_name'],
        }

        predicted_price = predict_price(input_data)
        return render_template('index.html', prediction=f"Predicted Price: ${predicted_price:.2f}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
