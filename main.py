import streamlit as st
import numpy as np
from joblib import load

st.title('Can you survive Titanic?')
st.markdown('_Imagine that you have boarded the Titanic in 1912, what would have been your fate?\
            This test allows for an 80% accuracy in determining whether you would have survived or not. \
            The data used to train this model was sourced from Kaggle. \
            Make by nicksttar_')

model = load('titanic_model.joblib')



def predict_survival(pclass, age, sibsp, parch, fare, sex_male, q, s):
    data = np.array([[pclass, age, sibsp, parch, fare, sex_male, q, s]])
    prediction =model.predict(data)
    proba = model.predict_proba(data)
    proba_a = str(proba[0][1] * 100)[:4]
    proba_na = str(proba[0][0] * 100)[:4]
    if prediction[0] == 0:
        return f'Ohhh, you would die in titanic if you was there with chance of {proba_na}%'
    else:
        return f'Congratulation you are alive with chance of {proba_a}% '
    
pclass = st.selectbox('Cabin class: ', [1, 2, 3])
age = st.slider('Age: ', 0, 100, step=1)
sibsp = st.selectbox('Have you parents on board: ', ['Yes', 'No'])
parch = st.selectbox('Have you girlfrien/boyfried on board: ', ['Yes', 'No'])
sex_male = st.selectbox('Gender: ', ['male', 'female'])
port = st.selectbox('Port of your boarding', ['Cherbourg', 'Queenstown', 'Southampton'])

if st.button("Predict"):
    fare = 32 # Mean of column Fare from data
    sibsp = (0, 1)[sibsp == 'Yes']
    parch = (0, 1)[parch == 'Yes']
    sex_male = (0, 1)[sex_male == 'male']
    if port == 'Cherbourg': 
        q = 0
        s = 0
    elif port == 'Queenstown':
        q = 1
        s = 0
    elif port == 'Southampton':
        q = 0
        s = 1
    result = predict_survival(pclass, age, sibsp, parch, fare, sex_male, q, s)
    st.write(result)