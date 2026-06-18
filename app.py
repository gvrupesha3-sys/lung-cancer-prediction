
import streamlit as st
import joblib  
import pandas as pd

# Load the trained model
model = joblib.load("lung_cancer_model.pkl")

# Feature names (same order used during training)
feature_names = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY",
    "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🫁 Model Information")
st.sidebar.write("**Algorithm:** Random Forest Classifier")
st.sidebar.write("**Accuracy:** 96.77%")
st.sidebar.write("**Dataset Size:** 309 Records")
st.sidebar.write("**Features Used:** 15")

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("🫁 Lung Cancer Prediction System")
st.write("Enter patient details and click **Predict**.")

# Helper function for Yes/No inputs
# Dataset uses 1 = No, 2 = Yes
def yes_no_input(label):
    return st.selectbox(
        label,
        [1, 2],
        format_func=lambda x: "No" if x == 1 else "Yes"
    )

# -----------------------------
# INPUTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    smoking = yes_no_input("Smoking")
    yellow_fingers = yes_no_input("Yellow Fingers")
    anxiety = yes_no_input("Anxiety")
    peer_pressure = yes_no_input("Peer Pressure")
    chronic_disease = yes_no_input("Chronic Disease")
    fatigue = yes_no_input("Fatigue")

with col2:
    allergy = yes_no_input("Allergy")
    wheezing = yes_no_input("Wheezing")
    alcohol_consuming = yes_no_input("Alcohol Consuming")
    coughing = yes_no_input("Coughing")
    shortness_of_breath = yes_no_input("Shortness of Breath")
    swallowing_difficulty = yes_no_input("Swallowing Difficulty")
    chest_pain = yes_no_input("Chest Pain")

# Convert gender to numeric
gender_value = 1 if gender == "Male" else 0

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔍 Predict", use_container_width=True):

    # Input data in the exact order used during training
    sample = [[
        gender_value,
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergy,
        wheezing,
        alcohol_consuming,
        coughing,
        shortness_of_breath,
        swallowing_difficulty,
        chest_pain
    ]]

    # Create DataFrame (recommended to avoid feature name warnings)
    sample_df = pd.DataFrame(sample, columns=feature_names)

    # Prediction
    prediction = model.predict(sample_df)[0]

    # Probability of class 1 (YES)
    probability = model.predict_proba(sample_df)[0][1]
    percentage = probability * 100

    # Show prediction result
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Lung Cancer Detected (YES)")
    else:
        st.success("✅ No Lung Cancer Detected (NO)")

    # Probability display
    st.metric("Probability of Lung Cancer", f"{percentage:.2f}%")

    # Progress bar
    st.progress(int(percentage))

    # Risk Level
    if percentage < 25:
        risk_level = "🟢 Low Risk"
    elif percentage < 50:
        risk_level = "🟡 Moderate Risk"
    elif percentage < 75:
        risk_level = "🟠 High Risk"
    else:
        risk_level = "🔴 Very High Risk"

    st.subheader(f"Estimated Risk Level: {risk_level}")

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    if hasattr(model, "feature_importances_"):
        st.subheader("📈 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.bar_chart(
            importance_df.set_index("Feature")
        )

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------
    report = f"""
LUNG CANCER PREDICTION REPORT
=============================

Prediction: {"YES" if prediction == 1 else "NO"}
Probability: {percentage:.2f}%
Risk Level: {risk_level}

Model: Random Forest Classifier
Accuracy: 96.77%
"""

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="lung_cancer_report.txt",
        mime="text/plain"
    )

    # Disclaimer
    st.info(
        "This system is for educational purposes only. "
        "It predicts the likelihood of lung cancer based on the training dataset "
        "and does not replace professional medical diagnosis."
    )