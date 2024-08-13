

from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

with open('target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

with open('loan_model.pkl','rb') as f:
    model=pickle.load(f)

with open('label_encoders.pkl','rb') as f:
    label_encoders=pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

@app.route('/')
def home():
    return render_template('one.html')

@app.route('/predict' , methods=['POST'])
def predict():
    if request.method=='POST':

        gender = request.form['Gender']  
        married = request.form['Married']  
        dependents = request.form['Dependents']
        education = request.form['Education'] 
        self_employed = request.form['Self_Employed']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        married_encoded = label_encoders['Married'].transform([married])[0]
        dependents_encoded = label_encoders['Dependents'].transform([dependents])[0]
        education_encoded = label_encoders['Education'].transform([education])[0]
        self_employed_encoded = label_encoders['Self_Employed'].transform([self_employed])[0]
        property_area_encoded = label_encoders['Property_Area'].transform([property_area])[0]

        features = np.array([[gender_encoded, married_encoded, education_encoded, self_employed_encoded,
                              applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, 
                              dependents_encoded, property_area_encoded]])
        
        features=scaler.transform(features)

        prediction = model.predict(features)
        prediction_result = target_encoder.inverse_transform(prediction)


        return render_template("two.html",prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)