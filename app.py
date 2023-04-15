from flask import Flask,render_template,request
app=Flask(__name__)

import pandas as pd
import numpy as np
import pickle
model=pickle.load(open('carPrice.pkl',"rb"))
car=pd.read_csv('cleanedCar.csv')

@app.route('/')
def index():
    name=sorted(car['name'].unique())
    company=sorted(car['company'].unique())
    company.insert(0,'Select Car Company')
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=sorted(car['fuel_type'].unique())
    
    return render_template("index.html",car_models=name,companies=company,years=year,fuel_types=fuel_type)
@app.route('/predict',methods=['POST'])
def  predict():
    company=request.form.get('company')
    name=request.form.get('car_model')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    kms_driven=request.form.get('kms_driven')
    print(company,name,year,fuel_type,kms_driven)
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([name,company,year,kms_driven,fuel_type]).reshape(1, 5)))
    print(prediction)
    return str(np.round(prediction[0],2))

if __name__=='__main__':
    app.run()