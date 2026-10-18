"""
Heart Disease Prediction - Streamlit App
=========================================

A minimal web interface around the trained scikit-learn Pipeline stored in
'heart_disease_model.pkl'.

That one file holds the imputers, the StandardScaler, the OneHotEncoder AND the
trained Support Vector Classifier, all fitted on the training split only in
'Heart_Disease_Prediction.ipynb'. Because the pre-processing travels with the
model, this app never has to encode or scale anything by hand - it just passes
the four raw values straight to the Pipeline, exactly as the notebook does.

Educational project only - not a medical diagnosis.
"""

import os
import pickle

import pandas as pd
import streamlit as st

MODEL_FILE = "heart_disease_model.pkl"

# The exact four attributes, in the exact order, the Pipeline was trained on
FEATURE_ORDER = ["cp", "thalach", "exang", "oldpeak"]

# Readable labels for the two coded attributes
CHEST_PAIN_TYPES = {
    1: "1 - Typical angina",
    2: "2 - Atypical angina",
    3: "3 - Non-anginal pain",
    4: "4 - Asymptomatic",
}
EXERCISE_ANGINA = {
    0: "0 - No",
    1: "1 - Yes",
}


@st.cache_resource
def load_model():
    """Load the trained Pipeline (pre-processing + SVC) from disk."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{MODEL_FILE}' was not found next to app.py.")
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the four clinical measurements below and the trained Support Vector "
    "Classifier will predict whether heart disease is indicated."
)

# ---- Load the model (and report clearly if that fails) ----
try:
    model = load_model()
except Exception as error:  # noqa: BLE001 - surface any loading problem to the user
    st.error(f"Could not load the trained model: {error}")
    st.info(
        f"Make sure '{MODEL_FILE}' is in the same folder as app.py and that the "
        "installed scikit-learn version matches requirements.txt."
    )
    st.stop()

# ---- Exactly four inputs, matching the four training attributes ----
with st.form("patient_form"):
    cp = st.selectbox(
        "Chest pain type (cp)",
        options=list(CHEST_PAIN_TYPES.keys()),
        format_func=lambda value: CHEST_PAIN_TYPES[value],
        index=3,
        help="The type of chest pain reported by the patient.",
    )
    thalach = st.slider(
        "Maximum heart rate achieved (thalach)",
        min_value=60,
        max_value=220,
        value=150,
        step=1,
        help="Highest heart rate reached during the exercise stress test, in beats per minute.",
    )
    exang = st.selectbox(
        "Exercise-induced angina (exang)",
        options=list(EXERCISE_ANGINA.keys()),
        format_func=lambda value: EXERCISE_ANGINA[value],
        index=0,
        help="Did chest pain appear during exercise?",
    )
    oldpeak = st.slider(
        "ST depression induced by exercise (oldpeak)",
        min_value=0.0,
        max_value=7.0,
        value=1.0,
        step=0.1,
        help="How far the ECG ST segment dropped during exercise relative to rest.",
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a one-row DataFrame shaped exactly like the training data.
    # No manual encoding or scaling - the saved Pipeline does all of it.
    user_input = pd.DataFrame(
        {
            "cp": [cp],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
        }
    )[FEATURE_ORDER]

    try:
        predicted_outcome = int(model.predict(user_input)[0])
    except Exception as error:  # noqa: BLE001 - keep the app usable on any failure
        st.error(f"Prediction failed: {error}")
    else:
        if predicted_outcome == 1:
            st.error("### DISEASE INDICATED")
            st.write(
                "Based on these four measurements, the model predicts a pattern "
                "consistent with heart disease."
            )
        else:
            st.success("### NO DISEASE INDICATED")
            st.write(
                "Based on these four measurements, the model does not predict a "
                "pattern consistent with heart disease."
            )

        with st.expander("See the values sent to the model"):
            st.dataframe(user_input, hide_index=True)

st.divider()
st.warning(
    "**Disclaimer:** This is an educational prediction system built for a machine "
    "learning assignment. It is **not** a medical diagnosis and must not be used "
    "to make health decisions. Always consult a qualified healthcare professional."
)
st.caption(
    "Model: Support Vector Classifier inside a scikit-learn Pipeline, trained on the "
    "UCI Heart Disease dataset, processed Cleveland subset (heart.csv). "
    "See Heart_Disease_Prediction.ipynb."
)
