# --------------------------------------------------
# Import Libraries
# --------------------------------------------------
import pickle
import joblib
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
from streamlit_lottie import st_lottie
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# --------------------------------------------------
# Home Page
# --------------------------------------------------
def Home():
    st.title("EMPLOYEE SALARY PREDICTION")
    st.markdown("<h3 style='color: #FFA31E;'>Welcome to the Employee Salary Prediction App!</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#FFA31E;'>Please enter your name and explore the options.</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #FFA31E;'>Use the sidebar to navigate different sections.</h3>", unsafe_allow_html=True)

    url = requests.get("https://lottie.host/303f66a1-0ee8-444d-b3a9-c760ed85f902/hJwxaX7tnc.json")
    url_json = dict()

    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error fetching Lottie URL")

    st_lottie(url_json)

# --------------------------------------------------
# Problem Statement and Data Description
# --------------------------------------------------
def Problem_Description():
    st.title("EMPLOYEE SALARY PREDICTION")
    st.markdown("<h1 style='color: #FFA31E;'>Problem Statement</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00EA87;'>Predicting Employee Salaries Based on Various Features</h3>", unsafe_allow_html=True)

    st.write("""
    Many companies want to understand how different factors affect employee salaries.
    This helps HR and management make informed decisions about hiring, compensation, and workforce planning.
    This project aims to predict an employee's salary using features like age, gender, education level, job title, and years of experience.
    """)

    st.subheader("Goal :")
    st.write("""
    To build a machine learning model that can accurately predict salaries.
    This can help companies design fair and competitive pay structures.
    """)

    st.markdown("<h3 style='color: #FFA31E;'>Data Description</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<span style='color:#00EA87'><b>Age</b></span> : Age of the employee", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Gender</b></span> : Gender of the employee", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Education Level</b></span> : Highest qualification (e.g., Bachelors, Masters, PhD)", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Job Title</b></span> : Job role or designation", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Years of Experience</b></span> : Total work experience in years", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Salary</b></span> : Target variable — annual salary in Rupees", unsafe_allow_html=True)

    with col2:
        image = Image.open("salary.png")  # Replace with your image file name
        st.image(image, width=700)

# --------------------------------------------------
# Data Overview using Pandas Profiling
# --------------------------------------------------
def pandas_profiling():
    with open('style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st.title("EMPLOYEE SALARY PREDICTION")
    st.markdown("<h3 style='color: #FFA31E;'>Data Overview using Pandas Profiling</h3>", unsafe_allow_html=True)

    df = pd.read_csv("Salary_Data.csv")

    profile = ProfileReport(df, title="Pandas Profiling Report for Salary Data")
    st_profile_report(profile)

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------

def predict_salary(EducationLevel, JobTitle, YearsOfExperience, Age, Gender):
    # Load encoders and model
    onehotencoder = joblib.load('onehotencoder.pkl')
    scaler = joblib.load('std_scaler.pkl')
    model = joblib.load('salary_model.pkl')

    # 1️⃣ Numerical data
    num_df = pd.DataFrame([[YearsOfExperience, Age]], columns=['Years of Experience', 'Age'])
    num_scaled = pd.DataFrame(scaler.transform(num_df), columns=num_df.columns)

    # 2️⃣ Categorical data
    cat_df = pd.DataFrame([[EducationLevel, JobTitle, Gender]],
                          columns=['Education Level', 'Job Title', 'Gender'])
    cat_encoded = pd.DataFrame(onehotencoder.transform(cat_df),columns=onehotencoder.get_feature_names_out(['Education Level', 'Job Title', 'Gender'])
)


    # 3️⃣ Combine like concating_numcat_df
    X_input = pd.concat([num_scaled, cat_encoded], axis=1)

    # 4️⃣ Align columns: match training exactly
    expected_cols = model.feature_names_in_
    missing_cols = set(expected_cols) - set(X_input.columns)
    for col in missing_cols:
        X_input[col] = 0  # Add missing dummy columns as 0

    # Ensure column order
    X_input = X_input[expected_cols]

    # 5️⃣ Predict
    predicted_salary = model.predict(X_input)[0]
    return predicted_salary


# --------------------------------------------------
# Prediction Page
# --------------------------------------------------
def prediction_page(name):
    with open('style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st.title("EMPLOYEE SALARY PREDICTION")

    EducationLevel = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'], index=1)
    JobTitle = st.selectbox("Job Title", ['Data Analyst', 'Data Scientist', 'Manager', 'Software Engineer'], index=1)
    YearsOfExperience = st.number_input("Years of Experience", value=2, min_value=0)
    Age = st.number_input("Age", value=25, min_value=18)
    Gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], index=0)

    if st.button("Predict Salary"):
        predicted_salary = predict_salary(EducationLevel, JobTitle, YearsOfExperience, Age, Gender)
        Result_page(name, predicted_salary)

# --------------------------------------------------
# Result Page
# --------------------------------------------------
def Result_page(name, predicted_salary):
    with open('style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st.subheader(f"Hello, {name}!")
    st.write("Based on the given inputs, your **predicted annual salary** is:")
    st.subheader(f"₹ {predicted_salary:,.2f}")
    st.write("This prediction is generated using the trained ML model.")

# --------------------------------------------------
# Main function with Sidebar
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="Employee Salary Prediction App",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open('style.css') as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    menu = ["Home", "Problem Statement", "Data Overview", "Salary Prediction"]
    choice = st.sidebar.selectbox("Go to", menu)

    st.sidebar.subheader("User Info")
    name = st.sidebar.text_input("Your Name", "")

    if choice == "Home":
        Home()
    elif choice == "Problem Statement":
        Problem_Description()
    elif choice == "Data Overview":
        pandas_profiling()
    elif choice == "Salary Prediction":
        prediction_page(name)

# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == '__main__':
    main()
