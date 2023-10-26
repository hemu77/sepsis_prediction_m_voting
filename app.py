import streamlit as st
import joblib
import numpy as np

# Load the VotingClassifier model
voting_classifier = joblib.load('sepsis_classifier.pkl')

st.title("Sepsis Detection App")

# Sidebar
st.sidebar.header("Sample Test Data")
# Define input fields for each feature. Adjust these based on your dataset.
age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
hospital_admission_time = st.sidebar.number_input("Hospital Admission Time (hours)", min_value=0, step=1)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=0, max_value=300, step=1)
systolic_blood_pressure = st.sidebar.number_input("Systolic Blood Pressure (mmHg)", min_value=0, max_value=300, step=1)
icu_los = st.sidebar.number_input("ICU Length of Stay (hours)", min_value=0, step=1)
# Add more input fields as needed

# Create a button to trigger classification
if st.sidebar.button("Classify"):
    # Prepare the input data
    input_data = np.array([[age, hospital_admission_time, heart_rate, systolic_blood_pressure, icu_los]])  # Adjust this for your dataset
    # Perform classification
    prediction = voting_classifier.predict(input_data)
    if prediction[0] == 1:
        st.sidebar.success("Sepsis Detected")
    else:
        st.sidebar.info("No Sepsis Detected")

# Main content
st.header("Sepsis Detection")
st.write("This app allows you to input patient data for sepsis detection.")
st.write("Enter the patient's data in the sidebar and click 'Classify' to get the result.")

# Example Testimonials
st.header("Sample Testimonials")
testimonials = {
    "Age": [65, 42, 35],
    "Hospital Admission Time (hours)": [2, 3, 4],
    "Heart Rate (bpm)": [90, 75, 110],
    "Systolic Blood Pressure (mmHg)": [120, 130, 140],
    "ICU Length of Stay (hours)": [24, 48, 72],
    "Sepsis": ["No", "No", "Yes"],
}

# Loop through the testimonials and classify them
for i in range(len(testimonials["Age"])):
    st.write("Testimonial", i + 1)
    st.write("Age:", testimonials["Age"][i])
    st.write("Hospital Admission Time (hours):", testimonials["Hospital Admission Time (hours)"][i])
    st.write("Heart Rate (bpm):", testimonials["Heart Rate (bpm)"][i])
    st.write("Systolic Blood Pressure (mmHg):", testimonials["Systolic Blood Pressure (mmHg)"][i])
    st.write("ICU Length of Stay (hours):", testimonials["ICU Length of Stay (hours)"][i])

    input_data = np.array([[
        testimonials["Age"][i],
        testimonials["Hospital Admission Time (hours)"][i],
        testimonials["Heart Rate (bpm)"][i],
        testimonials["Systolic Blood Pressure (mmHg)"][i],
        testimonials["ICU Length of Stay (hours)"][i]
    ]])
    
    # Perform classification
    prediction = voting_classifier.predict(input_data)
    if prediction[0] == 1:
        st.write("Classification: Sepsis Detected")
    else:
        st.write("Classification: No Sepsis Detected")
    st.write("")

# Conclusion
st.header("Conclusion")
st.write("This application demonstrates sepsis detection based on patient data.")
st.write("The classification results are displayed in the sidebar.")
