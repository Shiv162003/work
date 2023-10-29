import streamlit as st
from PIL import Image
import io
import base64
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
icon_image_url = 'assets/logo.jpg'  # Replace with the URL of your icon image
st.set_page_config(page_title="MediConnect ", layout="wide", page_icon=icon_image_url)

menu_options = ["Home", "Model","Meet Our  Team"]
selected = option_menu(menu_title=None, options=menu_options, orientation="horizontal")

# Define content for different sections based on the selected option
if selected == "Home":
    # Define content for the Home page
    st.title("About the Topic")

    # Left column with images
    col1, col2 = st.columns([1, 2])
    image_paths = ["assets/xx.jpg"]
    with col1:
        for image_path in image_paths:
            st.image(image_path, caption=f"Image {image_paths.index(image_path) + 1}", use_column_width=True)

    # Right column with text content
    with col2:


        # Set the title and subheader
        st.title("Specialist Recommendation and Machine Learning")
        st.subheader("Solving the Challenge of Finding the Right Specialist")

        # Introduction
        st.write("Finding the right specialist for your medical condition can be a daunting task, particularly in complex or unique situations. Here's why it can be challenging:")

        # Challenges List
        st.write("1. **Diverse Specialties**: The medical field is highly specialized, with numerous subspecialties. Finding the right specialist for a specific medical condition can be overwhelming.")
        st.write("2. **Limited Accessibility**: Specialists may not be readily accessible in all geographical areas, requiring patients to travel long distances.")
        st.write("3. **Complex Medical Histories**: Patients often have complex medical histories with multiple conditions, making it hard to determine which specialist is most appropriate.")
        st.write("4. **Insurance Constraints**: Insurance plans may have restrictions on which specialists are covered, limiting a patient's options.")
        st.write("5. **Referral Dependency**: Patients often rely on primary care physicians for referrals, which can be influenced by various factors.")

        # Machine Learning Benefits
        st.subheader("How Machine Learning Can Help:")
        st.write("Machine learning (ML) can significantly aid in addressing these challenges:")

        # ML Solutions List
        st.write("1. **Specialist Recommendation Systems**: ML-based recommendation systems can analyze a patient's medical history and symptoms to recommend the most suitable specialist.")
        st.write("2. **Geographical Insights**: ML can help determine the most accessible specialists based on a patient's location.")
        st.write("3. **Complex Case Analysis**: ML can analyze complex medical histories and identify patterns that may not be evident to humans.")
        st.write("4. **Insurance Compatibility**: ML can consider insurance plan constraints and recommend specialists that are covered by the patient's insurance.")
        st.write("5. **Data-Driven Referrals**: Primary care physicians can benefit from ML tools that provide data-driven recommendations based on historical outcomes and patient preferences.")

        # Conclusion
        st.subheader("Conclusion")
        st.write("Machine learning, combined with electronic health records (EHR) and patient data, has the potential to transform the way specialist recommendations are made. These systems can continuously learn and improve their recommendations, ensuring that patients receive the most appropriate care based on their unique situations. They also have the potential to reduce misdiagnoses and improve overall healthcare outcomes.")


        
        
        
        
        
        
        
        
        
if selected == "Model":
    label_mapping = {
    1: "Dermatologist",
    2: "Allergist",
    3: "Gastroenterologist",
    4: "Physician",
    5: "Osteopathic",
    6: "Endocrinologist",
    7: "Pulmonologist",
    8: "Cardiologist",
    9: "Neurologist",
    10: "Internal Medcine",
    11: "Pediatrician",
    12: "Common Cold",
    13: "Cardiologist",
    14: "Phlebologist",
    15: "Osteoarthristis",
    16: "Rheumatologists",
    17: "Otolaryngologist",
    18: "Dermatologists",
    19: "Gynecologist"
    }
    df = pd.read_csv("output.csv")

    # Split the dataset into features (X) and the target variable (y)
    X = df.iloc[:, :-1]
    y = df['doc_num']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    # Predict the target variable on the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Random Forest Classifier: {accuracy * 100:.2f}%")
    symptoms = [
        "itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches",
        "continuous_sneezing", "shivering", "chills", "watering_from_eyes",
        "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough",
        "chest_pain", "yellowish_skin", "nausea", "loss_of_appetite",
        "abdominal_pain", "yellowing_of_eyes", "burning_micturition",
        "spotting_urination", "passage_of_gases", "internal_itching", "indigestion",
        "muscle_wasting"
    ]

    # Create a Streamlit web application
    st.title("Symptom Checker")

    st.write("Please answer the following questions with 'Yes' or 'No':")

    # Initialize an empty dictionary to store the user's responses
    user_symptoms = {}

    # Ask the user about each symptom and record their responses
    for symptom in symptoms:
        response = st.radio(f"Do you have {symptom}?", ["Yes", "No"])
        if response == "Yes":
            user_symptoms[symptom] = 1
        else:
            user_symptoms[symptom] = 0

    # Add a submit button
    if st.button("Submit"):
        # Load your trained Random Forest classifier (replace 'your_model.pkl' with the actual model file)
        

        # Convert user responses into a DataFrame for prediction
        user_data = pd.DataFrame([user_symptoms])

        # Make predictions with the model
        predictions = rf_classifier.predict(user_data)
        mapped_predictions = [label_mapping[pred] for pred in predictions]
        # Display the model's predictions
        st.write("Specialist_recommended is :")
        x=mapped_predictions
        st.write(x)

    
    
    
    
if selected=="Meet Our  Team":
    # Function to display team member information
    def display_team_member(image_filename, name, roll_number, email):
        st.markdown(
            f"## {name}",
            unsafe_allow_html=True
        )
        st.image(f"assets/{image_filename}")
        st.write(f"Roll Number: {roll_number}")
        st.write(f"Email: {email}")
    # Function to display mentor information
    def display_mentor(image_filename, name, linkedin_profile):
        st.markdown(f"## {name}", unsafe_allow_html=True)
        st.image(f"assets/{image_filename}")
        st.write(f"LinkedIn Profile: [{name}]({linkedin_profile})")
    col1, col2,col3,col4,col5 = st.columns(5)
    with col3: 
        display_team_member("m.png", "Hitiksh Doshi", "21070124504", "Hitiksh.Doshi.btech2020@sitpune.edu.in")


        
    col1, col2,col3,col4,col5 = st.columns(5)
    
    with col3:
        # Bolder text using HTML and CSS
        st.markdown("<p style='font-size: 40px; font-weight: bold; color: red;'>TEAM</p>", unsafe_allow_html=True)

        
    col1, col2, col3,col4,col5 = st.columns(5)
    
    # Display team members' information in each column
    with col2:
        display_team_member("1.png", "Ahmed Ibrahim Siddiqui", "20070124009", "Ahmed.Ibrahim.btech2020@sitpune.edu.in")

    with col3:
        display_team_member("2.png", "Shah Dev Sanjay", "21070124502", "Shah.Dev.btech2020@sitpune.edu.in")

    with col4:
        display_team_member("3.png", "Ujjas  Thacker", "20070124038", "Ujjas.Thacker.btech2020@sitpune.edu.in")
