from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.flight_pipeline.flight_predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect form inputs (ensure the HTML <form> field names match these exactly)
        data = CustomData(
            from_=request.form.get('from'),
            to=request.form.get('to'),
            flightType=request.form.get('flightType'),
            time=float(request.form.get('time')),
            distance=float(request.form.get('distance')),
            agency=request.form.get('agency'),
            date=request.form.get('date')   # Should be "MM/DD/YYYY"
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        # Display the result in your template
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
