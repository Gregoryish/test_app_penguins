import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

clf = pickle.load(open('penguins_classifier.sav', 'rb'))


st.write("""
# This is PenguinsApp based on Streamlit and ML-Classifier
 *gregoryish@yandex.ru*
""")
st.subheader("see sidebar and ***set your inputs***  to determine penguins **specious**")
st.sidebar.header("Set inputs")
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/Gregoryish/data/master/penguins_example.csv)""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length, mm', 32, 60, 40)
        bill_depth_mm = st.sidebar.slider('Bill depth, mm', 13, 22, 17)
        flipper_lenght_mm = st.sidebar.slider('Flipper length, mm', 172, 231, 201)
        body_mass_g = st.sidebar.slider('Body mass, g', 2700, 6300, 4200)

        data = {'island': island,
                'sex': sex,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_lenght_mm,
                'body_mass_g': body_mass_g}

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


df = pd.read_csv(r'D:\GregoryishGithub\data\penguins_cleaned.csv')
targets = df['species'].unique()
dict_target = dict(zip([1, 2, 3], targets))
df = df.drop('species', axis=1)
df = pd.concat([input_df, df], axis=0)

get_dummies = ['island', 'sex']

for col in get_dummies:
    df = pd.concat([df, pd.get_dummies(df[col])], axis=1)
    df = df.drop(col, axis=1)
df = df[:1]
st.subheader('User Input features')
st.write(df)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction')
st.write(np.array([dict_target[prediction[0]]]))

st.subheader('Prediction Probabilities')

df_proba = pd.DataFrame({'Species': targets,
                    'Probabilities': prediction_proba[0]*100})

st.write(df_proba)