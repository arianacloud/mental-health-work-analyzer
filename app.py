
import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

st.set_page_config(page_title="Mental Health Risk Analyzer", layout="wide")

@st.cache_resource
def load_models():
    model_full = joblib.load("model_full.pkl")
    model_hr = joblib.load("model_hr.pkl")
    with open("feature_names.json") as f:
        feature_names = json.load(f)
    with open("feature_values.json") as f:
        feature_values = json.load(f)
    return model_full, model_hr, feature_names, feature_values

model_full, model_hr, feature_names, feature_values = load_models()

st.title("Mental Health Work Interference Analyzer")
st.markdown("### Helping HR teams understand and reduce work interference risk")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["HR Policy Tool", "Research Insights", "What Ethics Limits"])

with tab1:
    st.subheader("Company Policy Settings")
    st.markdown("*Adjust your company policies to see how they affect work interference risk*")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Benefits & Support**")
        benefits = st.selectbox("Mental Health Benefits", feature_values["benefits"])
        care_options = st.selectbox("Care Options Available", feature_values["care_options"])
        wellness_program = st.selectbox("Wellness Program", feature_values["wellness_program"])
        seek_help = st.selectbox("Resources to Seek Help", feature_values["seek_help"])
        anonymity = st.selectbox("Anonymity Protected", feature_values["anonymity"])

    with col_b:
        st.markdown("**Policies & Culture**")
        leave = st.selectbox("Ease of Mental Health Leave", feature_values["leave"])
        mental_health_consequence = st.selectbox("Consequence of Disclosing Mental Health", feature_values["mental_health_consequence"])
        phys_health_consequence = st.selectbox("Consequence of Disclosing Physical Health", feature_values["phys_health_consequence"])
        mental_vs_physical = st.selectbox("Mental Health Treated Like Physical", feature_values["mental_vs_physical"])
        obs_consequence = st.selectbox("Observed Negative Consequences", feature_values["obs_consequence"])

    with col_c:
        st.markdown("**Team & Company**")
        coworkers = st.selectbox("Discuss with Coworkers", feature_values["coworkers"])
        supervisor = st.selectbox("Discuss with Supervisor", feature_values["supervisor"])
        no_employees = st.selectbox("Company Size", feature_values["no_employees"])
        remote_work = st.selectbox("Remote Work", feature_values["remote_work"])
        tech_company = st.selectbox("Tech Company", feature_values["tech_company"])

    st.markdown("---")

    hr_input = pd.DataFrame([{
        "benefits": benefits, "care_options": care_options,
        "wellness_program": wellness_program, "seek_help": seek_help,
        "anonymity": anonymity, "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers, "supervisor": supervisor,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence, "no_employees": no_employees,
        "remote_work": remote_work, "tech_company": tech_company
    }])

    hr_prob = float(model_hr.predict_proba(hr_input)[0][1])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Score")
        if hr_prob > 0.65:
            st.error(f"Work Interference Probability: {hr_prob:.0%} — High Risk")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Review mental health leave policies")
            st.markdown("- Implement wellness programs")
            st.markdown("- Ensure anonymity in mental health discussions")
        elif hr_prob > 0.45:
            st.warning(f"Work Interference Probability: {hr_prob:.0%} — Medium Risk")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Consider adding mental health benefits")
            st.markdown("- Train supervisors on mental health support")
        else:
            st.success(f"Work Interference Probability: {hr_prob:.0%} — Low Risk")
            st.markdown("Good policies in place. Keep maintaining these standards.")

    with col2:
        st.subheader("Key HR Factors")
        st.markdown("🔴 Making mental health leave difficult → High Risk")
        st.markdown("🔴 No wellness program → High Risk")
        st.markdown("🔴 Fear of mental health disclosure → High Risk")
        st.markdown("🔴 No resources to seek help → High Risk")
        st.markdown("🟢 No fear of disclosure → Protective")
        st.markdown("🟢 Care options available → Protective")
        st.markdown("🟢 Anonymity protected → Protective")

with tab2:
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.info("Full Model (all factors) | AUC: 0.760 | F1: 0.785")
    with col2:
        st.warning("HR Model (workplace only) | AUC: 0.633 | F1: 0.669")
    st.markdown("---")
    st.subheader("What drives the gap?")
    st.markdown("The 0.127 AUC gap between models shows that personal factors like treatment history are strong predictors — but HR cannot and should not access them. Workplace policies alone partially predict work interference; the rest depends on individual circumstances that must remain private.")
    st.markdown("---")
    st.subheader("Top HR-Controllable Risk Factors")
    st.markdown("🔴 Difficult mental health leave")
    st.markdown("🔴 No wellness program")
    st.markdown("🔴 Fear of disclosure consequences")
    st.markdown("🔴 No help-seeking resources")
    st.markdown("🟢 No fear of disclosure")
    st.markdown("🟢 Care options available")
    st.markdown("🟢 Anonymity protected")

with tab3:
    st.subheader("What the data reveals — but ethics prevents HR from using")
    st.markdown("The full model identified two highly predictive personal factors. Here is what the data shows, and why HR should not act on it directly.")
    st.markdown("---")
    data = {
        "Factor": ["No treatment history", "Family history of mental illness", "Very difficult leave policy", "No wellness program", "Fear of disclosure"],
        "Impact on Risk": ["Very High", "High", "Moderate", "Moderate", "Moderate"],
        "HR Can Use?": ["No — Privacy", "No — Privacy", "Yes", "Yes", "Yes"],
        "Why": ["Medical history is private by law", "Family medical history is private", "Leave policy is an HR decision", "Wellness programs are HR decisions", "Culture change is within HR control"]
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True)
    st.markdown("---")
    st.subheader("The Right Conclusion")
    st.success("Employees who have not sought treatment are at significantly higher risk. The solution is NOT for HR to ask about treatment history. The solution is building an environment where employees voluntarily seek help through accessible care options, protected anonymity, and a culture where mental health is treated like physical health.")

    st.markdown("---")
    st.subheader("The Numbers Behind the Story")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Treatment History**")
        st.error("Having sought treatment: +40% correlation with work interference")
        st.success("No treatment history: -40% correlation with work interference")
        st.markdown("*Strongest predictor in the entire dataset*")
    with col2:
        st.markdown("**Family History**")
        st.error("Family history of mental illness: +23% correlation with work interference")
        st.success("No family history: -23% correlation with work interference")
        st.markdown("*Second strongest predictor after treatment*")
    st.warning("These two factors explain most of the 0.127 AUC gap between Full Model (0.760) and HR Model (0.633). The most predictive information is also the most sensitive.")

    st.markdown("---")
    st.info("Privacy Note: This tool only uses workplace policy factors HR can ethically observe and control.")
    st.caption("Based on OSMI Mental Health in Tech Survey 2014 | HR Model AUC: 0.633 | Full Model AUC: 0.760")

