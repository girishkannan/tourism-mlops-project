"""Streamlit app: predict whether a customer will buy the Wellness Tourism Package."""
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
THRESHOLD = 0.45


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("Wellness Tourism Package Purchase Predictor")
st.write("Enter customer details to predict whether they are likely to purchase the package.")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Number of Trips per Year", 0, 25, 3)
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
with col2:
    own_car = st.selectbox("Owns Car?", ["Yes", "No"])
    num_children = st.number_input("Children Visiting (below 5)", 0, 5, 1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 20000, step=500)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    pitch_duration = st.number_input("Duration of Pitch (minutes)", 1, 150, 15)

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": pitch_duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0, 1]
    if prob >= THRESHOLD:
        st.success(f"Likely to purchase the package (probability {prob:.1%}).")
    else:
        st.warning(f"Unlikely to purchase the package (probability {prob:.1%}).")
