import numpy as np
from flask import Flask, request, jsonify, render_template, make_response
import pickle
import xgboost
import pandas as pd
import sklearn

app = Flask(__name__)
model = pickle.load(open('./model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """"Rendering results in the Web App"""
    f = request.files['csv_file']
    if request.method == 'POST' and f.filename != '':
        f = request.files['csv_file']
        df_raw = pd.read_csv(f)
        if transform_input(df_raw) is False:
            return None  # HOW TO SHOW ERROR IN FLASK
        else:
            x = transform_input(df_raw)
            y = model.predict(x)

            # convert to dataframe and allow user to download
            df_y = pd.DataFrame(y)
            output = make_response(df_y.to_csv())
            output.headers["Content-Disposition"] = "attachment; filename=export.csv"
            output.headers["Content-type"] = "text/csv"
            return output

    return render_template('index.html', prediction_text='Please upload a correct .csv file!')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    """Direct API requests"""
    data = request.get_json(force=True)
    x = transform_input
    pass
    # NEED TO BUILD API

def transform_input(df_input):
    """Used to transform .csv file data for input into model"""
    """If returns False, means that not appropriate data"""

    df_output = df_input.copy()

    # drop variables
    drop_labels = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18', 'YearsAtCompany', 'JobLevel']
    df_output.drop(labels=drop_labels, axis=1, inplace=True)

    # encode variables
    categorical_data = df_output.dtypes.loc[df_output.dtypes == 'object'].index
    categorical_data = list(categorical_data)
    categorical_data.remove('BusinessTravel')

    business_encode_mask = {'BusinessTravel': {'Travel_Frequently': 2, 'Travel_Rarely': 1, 'Non-Travel': 150}}
    df_output.replace(business_encode_mask, inplace=True)

    df_output = pd.get_dummies(df_output, columns=categorical_data, drop_first=True)

    # create columns if they don't exist
    std_columns = set(
        {'Age', 'BusinessTravel', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
         'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
         'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
         'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
         'YearsWithCurrManager', 'Department_Research & Development', 'Department_Sales',
         'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other',
         'EducationField_Technical Degree', 'Gender_Male', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
         'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist',
         'JobRole_Sales Executive', 'JobRole_Sales Representative', 'MaritalStatus_Married', 'MaritalStatus_Single',
         'OverTime_Yes'})

    # there is probably a better work around to the above, but it helps for the categorical encoding field
    # if the categories don't exist

    curr_columns = set(df_output.columns)

    col_to_create = std_columns.difference(curr_columns)

    for col in col_to_create:
        df_output[col] = 0

    df_output = sklearn.preprocessing.normalize(df_output)
    return df_output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8080)
    # app.run(debug=True)
