import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import StandardScaler

st.title('Titanic Survival Prediction')
st.markdown('_Imagine that you have boarded the Titanic in 1912, what would have been your fate? '
            'This test allows for an 80% accuracy in determining whether you would have survived or not. '
            'The data used to train this model was sourced from Kaggle. '
            'Made by nicksttar._')


def predict_survival(pclass: int, age: int, sibsp: str, parch: str, fare: int, sex_male: str, q: int, s: int) -> str:
    """
    This function takes in various features related to a passenger on the Titanic and predicts 
    whether they would survive or not based on the given model. The features include the passenger's 
    cabin class, age, number of siblings/spouses aboard, number of parents/children aboard, fare, 
    gender, and the port of embarkation. The function returns a string with the predicted outcome 
    and the probability of survival in percentage.

    Args:
    - pclass: int, the cabin class of the passenger (1, 2, or 3).
    - age: int, the age of the passenger.
    - sibsp: str, 'Yes' or 'No', whether the passenger has parents aboard.
    - parch: str, 'Yes' or 'No', whether the passenger has wife/husband aboard.
    - fare: float, the fare paid by the passenger.
    - sex_male: str, 'male' or 'female', the gender of the passenger.
    - q: int, binary variable indicating whether the passenger embarked from Queenstown.
    - s: int, binary variable indicating whether the passenger embarked from Southampton.

    Returns:
    - A string with the predicted outcome and the probability of survival in percentage.
    """
    # Load the scaler and data used to scale the features
    model = load('base_models/titanic_model.joblib')

    # Scaling
    X_need_to_scale = pd.read_csv('base_models/X_need_to_scale.csv')
    scaler = StandardScaler()
    scaler.fit(X_need_to_scale)

    # Take data and transform it
    data = [[pclass, age, sibsp, parch, fare, sex_male, q, s]]
    data = scaler.transform(data)

    # Make prediction and probabillity
    prediction = model.predict(data)
    proba = model.predict_proba(data)
    proba_a = str(proba[0][1] * 100)[:4]
    proba_na = str(proba[0][0] * 100)[:4]
    if prediction[0] == 0:
        return f'Ohhh, you would die in Titanic if you were there with chance of {proba_na}%'
    else:
        return f'Congratulations, you are alive with chance of {proba_a}%'

def string_to_digit(sibsp: str, parch: str, sex_male: str, port: str) -> list:
    """
    This function takes in four string arguments representing various features related to a passenger on the Titanic. 
    Sibsp, parch, sex_male, and port. It converts these features to numeric values and returns a list containing these values.
    """
    sibsp = int(sibsp == 'Yes')
    parch = int(parch == 'Yes')
    sex_male = int(sex_male == 'male')
    port_map = {'Cherbourg': (0, 0), 'Queenstown': (1, 0), 'Southampton': (0, 1)}
    q, s = port_map.get(port, (0, 0))
    return [sibsp, parch, sex_male, q, s]

def predict_fare(pclass: int, age: int, sibsp: str, parch: str, sex_male: str, port: str) -> float:
    # Convert categorical variables to numerical
    convert = string_to_digit(sibsp, parch, sex_male, port)
    
    # Load data and model
    X = pd.read_csv('base_models/X_need_to_fare_scale.csv')
    model = load('base_models/fare_model.joblib')
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Prepare the data for prediction
    data = [[pclass, age, convert[0], convert[1], convert[2], convert[-2], convert[-1]]]
    data = scaler.transform(data)

    # Make the prediction
    prediction = model.predict(data)
    fare = prediction[0]

    # Adjust fare for first-class passengers
    if pclass == 1:
        fare += 30  # From tests we have seen that the average fare for first-class passengers is 30 higher than the predicted fare

    return float(fare)
    

pclass = st.selectbox('Cabin class: ', [1, 2, 3])
age = st.slider('Age: ', 0, 100, step=1)
sibsp = st.selectbox('Have you parents on board: ', ['Yes', 'No'])
parch = st.selectbox('Have you girlfriend/boyfriend on board: ', ['Yes', 'No'])
sex_male = st.selectbox('Gender: ', ['male', 'female'])
port = st.selectbox('Port of your boarding', ['Cherbourg', 'Queenstown', 'Southampton'])

if st.button("Predict"):
    fare = predict_fare(pclass, age, sibsp, parch, sex_male, port)
    convert = string_to_digit(sibsp, parch, sex_male, port)
    result = predict_survival(pclass, age, convert[0], convert[1], fare, convert[2], convert[-2], convert[-1])
    st.write(result)
    