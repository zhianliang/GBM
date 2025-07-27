import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="Popliteal Artery Runoff Prediction", layout="wide")
st.title("Popliteal Artery Runoff Prediction Model")
st.sidebar.header("Input Parameters")

# Input widgets
degree_of_popliteal_artery_stenosis = st.sidebar.slider(
    "Degree of Popliteal Artery Stenosis (1-4)",
    min_value=1, max_value=4, value=2, step=1,
    help="1: 0-49%, 2: 50-69%, 3: 70-99%, 4: 100%"
)

degree_of_infrapopliteal_arteries_stenosis = st.sidebar.slider(
    "Degree of Infrapopliteal Arteries Stenosis (1-4)",
    min_value=1, max_value=4, value=2, step=1,
    help="1: 0-49%, 2: 50-69%, 3: 70-99%, 4: 100%"
)

Rutherford = st.sidebar.slider(
    "Rutherford Classification (1-6)",
    min_value=1, max_value=6, value=3, step=1,
    help="1: Mild claudication, 2: Moderate claudication, 3: Severe claudication, 4: Rest pain, 5: Moderate Tissue loss, 6: Severe Tissue loss"
)

ABI = st.sidebar.slider(
    "Ankle-Brachial Index (ABI)",
    min_value=0.0, max_value=1.5, value=0.8, step=0.01,
    help="Normal range: 0.9-1.3"
)

TcPO2 = st.sidebar.slider(
    "Transcutaneous Oxygen Pressure (TcPO2, mmHg)",
    min_value=0.0, max_value=100.0, value=40.0, step=0.1,
    help="Normal value: >60 mmHg"
)

eGFR = st.sidebar.slider(
    "Estimated Glomerular Filtration Rate (eGFR, mL/min/1.73m²)",
    min_value=0.0, max_value=150.0, value=60.0, step=0.1,
    help="Normal value: ≥90"
)


# Load model with caching
@st.cache_resource
def load_model():
    return joblib.load('gbm.pkl')


model = load_model()

# Prepare input data
input_data = pd.DataFrame({
    'degree_of_popliteal_artery_stenosis': [degree_of_popliteal_artery_stenosis],
    'degree_of_infrapopliteal_arteries_stenosis': [degree_of_infrapopliteal_arteries_stenosis],
    'Rutherford': [Rutherford],
    'ABI': [ABI],
    'TcPO2': [TcPO2],
    'eGFR': [eGFR]
})

# Prediction button
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(input_data)

    # Display results
    st.subheader("Prediction Result")
    result = "Low Risk" if prediction[0] == 0 else "High Risk"
    st.write(f"Predicted outcome: **{result}**")

    # SHAP explanation
    st.subheader("Model Explanation (SHAP Values)")
    st.write("The force plot below shows how each feature contributes to the prediction:")

    try:
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Create force plot
        plt.figure(figsize=(10, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_data.iloc[0],
            matplotlib=True,
            show=False,
            text_rotation=15
        )
        st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")

# Instructions
st.sidebar.markdown("""
### Instructions
1. Adjust parameters using the sliders
2. Click the "Predict" button
3. View results and explanation in the main panel
""")