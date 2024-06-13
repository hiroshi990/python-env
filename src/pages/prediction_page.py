import streamlit as st 
import pickle
import pandas as pd 
import numpy as np
from src.utils import load_object



model_pipeline=load_object("model.pkl")
transformer=load_object("preprocessor.pkl")


def predict_page():
    
    gender_opt=["male","female"]
    race_ethnicity_opt=['group B','group C','group A','group D','group E']
    parent_education_opt=["bachelor's degree","master's degree","associate's degree",'high school']
    lunch_opt=["standard","free/reduced"]
    course_opt=['none','completed']
    
    
    st.title("Student Performance Predictor")
    st.write('''Fill the following fields below''')
    gender=st.selectbox("Gender",gender_opt)
    race_ethnicity=st.selectbox("Race",race_ethnicity_opt)
    parental_level_of_education=st.selectbox("Parental Level of Education",parent_education_opt)
    lunch=st.selectbox("Lunch",lunch_opt)
    test_preparation_course=st.selectbox("Preparation Course",course_opt)
    writing_score=st.number_input("Enter Writing Score",0,100)
    reading_score=st.number_input("Enter Reading Score",0,100)
    ok=st.button("Calculate Maths Score")
    if ok:
        # Prepare the input data as a dictionary
        data = {
            "gender": [gender],
            "race_ethnicity": [race_ethnicity],
            "parental_level_of_education": [parental_level_of_education],
            "lunch": [lunch],
            "test_preparation_course": [test_preparation_course],
            "writing_score": [writing_score],
            "reading_score": [reading_score]
        }

        # Convert the data dictionary to a DataFrame
        X = pd.DataFrame(data)

        # Preprocess the input data
        preprocessed_X = transformer.transform(X)

        # Make a prediction
        prediction = model_pipeline.predict(preprocessed_X)

        st.write(f"Predicted Math Score: {prediction[0]:.2f}")
        
        
    
    
    
    


    
    


    


