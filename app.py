import streamlit as st
import pandas as pd
import plotly.express as px
import traceback
import io

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================================================
# HIREWISE AI - JOB PREDICTION SYSTEM
# ==================================================

st.set_page_config(
    page_title="HireWise AI",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        color: #1f77b4;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: bold;
    }
    div[data-testid="stNotification"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 HireWise AI")
st.subheader("Human Job Search and Recruitment Outcome Prediction")
st.write(
    "Predict job search responses, interview results, "
    "and application status using Random Forest Classification."
)
st.divider()

# ==================================================
# 0. DATASET UPLOAD SECTION (Sidebar)
# ==================================================
st.sidebar.header("📁 Data Source")

# தேவையான டேட்டாசெட் பார்மெட் (Template) டவுன்லோட் செய்ய
sample_data = {
    "Education": ["BTech", "MCA", "BSc"],
    "Experience_Years": [2, 1, 0],
    "Skills_Score": [85, 70, 60],
    "Applications_Sent": [20, 15, 10],
    "Coding_Score": [90, 65, 50],
    "Communication_Score": [80, 75, 70],
    "Interview_Attended": [2, 1, 0],
    "Job_Search_Status": ["Good Response", "No Response", "Some Response"],
    "Interview_Result": ["Selected", "Rejected", "Pending"],
    "Application_Status": ["Approved", "Rejected", "Pending"]
}
template_df = pd.DataFrame(sample_data)
csv_template = template_df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="📥 Download Dataset Format (CSV)",
    data=csv_template,
    file_name='hirewise_dataset_format.csv',
    mime='text/csv',
)
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload your custom CSV Dataset", type=["csv"])

try:
    # ==================================================
    # 1. LOAD DATASET (With Fallback Safety)
    # ==================================================
    HF_CSV_URL = "https://huggingface.co/datasets/saroa/hirewise-dataset/resolve/main/human_job_prediction_50000_records_converted.csv"

    @st.cache_data
    def load_data(url, uploaded_file):
        required_columns = [
            "Education", "Experience_Years", "Skills_Score", "Applications_Sent",
            "Coding_Score", "Communication_Score", "Interview_Attended",
            "Job_Search_Status", "Interview_Result", "Application_Status"
        ]
        
        # 1. முதலில் அப்லோட் செய்யப்பட்ட பைலை செக் செய்வோம்
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                data.columns = data.columns.astype(str).str.strip()
                
                # தேவையான காலம்கள் உள்ளதா என்று பார்ப்போம்
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.error(f"⚠️ Invalid Uploaded Dataset! Missing columns: {', '.join(missing_columns)}. Switching to default dataset...")
                else:
                    st.success("✅ Custom dataset format is correct! Using uploaded data.")
                    # Clean and process the uploaded data
                    data["Application_Status"] = (
                        data["Application_Status"].astype("string").str.strip().replace({
                            "Selected": "Approved", "selected": "Approved", "Approved": "Approved",
                            "Not Selected": "Rejected", "Not selected": "Rejected",
                            "Rejected": "Rejected", "Rejected Application": "Rejected"
                        })
                    )
                    for column in ["Job_Search_Status", "Interview_Result", "Application_Status"]:
                        data[column] = data[column].astype("string").str.strip().replace(["", "nan", "None", "<NA>"], pd.NA)
                    for column in ["Experience_Years", "Skills_Score", "Applications_Sent", "Coding_Score", "Communication_Score", "Interview_Attended"]:
                        data[column] = pd.to_numeric(data[column], errors="coerce")
                    data["Education"] = data["Education"].astype("string").str.strip().replace(["", "nan", "None", "<NA>"], pd.NA)
                    return data
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}. Switching to default dataset...")

        # 2. அப்லோட் செய்தது தவறு என்றால், Default HF Dataset-ஐ லோட் செய்வோம்
        data = pd.read_csv(url)
        data.columns = data.columns.astype(str).str.strip()
        normalized = {
            str(column).strip().lower().replace(" ", "_").replace("-", "_"): column
            for column in data.columns
        }
        aliases = {
            "Education": ["education", "education_level", "qualification"],
            "Experience_Years": ["experience_years", "experience", "years_experience", "work_experience"],
            "Skills_Score": ["skills_score", "skill_score", "skills_rating", "skill_rating"],
            "Applications_Sent": ["applications_sent", "applications", "total_applications", "jobs_applied"],
            "Coding_Score": ["coding_score", "coding_test_score", "technical_score", "technical_test_score"],
            "Communication_Score": ["communication_score", "communication_test_score", "communication_rating"],
            "Interview_Attended": ["interview_attended", "interviews_attended", "interview_count", "number_of_interviews"],
            "Job_Search_Status": ["job_search_status", "job_response_status", "response_status", "job_search_outcome"],
            "Interview_Result": ["interview_result", "interview_status", "interview_outcome", "interview_decision"],
            "Application_Status": ["application_status", "job_selection_status", "selection_status", "selection_result", "job_status", "hiring_status", "final_status", "job_outcome", "recruitment_status", "application_result"],
        }
        rename_map = {}
        for standard_name, possible_names in aliases.items():
            if standard_name in data.columns: continue
            for possible_name in possible_names:
                actual_name = normalized.get(possible_name)
                if actual_name is not None:
                    rename_map[actual_name] = standard_name
                    break
        if rename_map: data = data.rename(columns=rename_map)

        # Clean default data
        data["Application_Status"] = (
            data["Application_Status"].astype("string").str.strip().replace({
                "Selected": "Approved", "selected": "Approved", "Approved": "Approved",
                "Not Selected": "Rejected", "Not selected": "Rejected",
                "Rejected": "Rejected", "Rejected Application": "Rejected"
            })
        )
        for column in ["Job_Search_Status", "Interview_Result", "Application_Status"]:
            data[column] = data[column].astype("string").str.strip().replace(["", "nan", "None", "<NA>"], pd.NA)
        for column in ["Experience_Years", "Skills_Score", "Applications_Sent", "Coding_Score", "Communication_Score", "Interview_Attended"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data["Education"] = data["Education"].astype("string").str.strip().replace(["", "nan", "None", "<NA>"], pd.NA)
        return data

    with st.spinner("Dataset Load ஆகிறது... தயவுசெய்து காத்திருங்கள்."):
        df = load_data(HF_CSV_URL, uploaded_file)

    if df.empty:
        st.error("The dataset is empty. Please check the CSV file.")
        st.stop()

    st.success(f"Dataset loaded successfully: {len(df):,} records")

    # ==================================================
    # 2. INPUT AND OUTPUT COLUMNS
    # ==================================================
    feature_columns = ["Education", "Experience_Years", "Skills_Score", "Applications_Sent", "Coding_Score", "Communication_Score", "Interview_Attended"]
    target_columns = ["Job_Search_Status", "Interview_Result", "Application_Status"]
    target_names = {"Job_Search_Status": "Job Search Status", "Interview_Result": "Interview Result", "Application_Status": "Application Status"}
    categorical_features = ["Education"]
    numeric_features = ["Experience_Years", "Skills_Score", "Applications_Sent", "Coding_Score", "Communication_Score", "Interview_Attended"]

    # ==================================================
    # 3. TRAIN RANDOM FOREST MODELS
    # ==================================================
    @st.cache_resource
    def train_models(data, data_source_key):
        models = {}
        metrics = {}
        for target in target_columns:
            model_data = data[feature_columns + [target]].dropna(subset=[target]).copy()
            model_data = model_data[model_data[target].astype("string").str.strip().ne("")]

            if model_data.empty: raise ValueError(f"No valid target values for {target}.")
            if model_data[target].nunique() < 2: raise ValueError(f"{target} needs at least two classes.")
            if model_data[target].value_counts().min() < 2: raise ValueError(f"{target} needs at least two records per class.")

            if len(model_data) > 70000:
                model_data = model_data.sample(n=70000, random_state=42)

            X = model_data[feature_columns]
            y = model_data[target].astype(str)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("education", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
                    ("numbers", SimpleImputer(strategy="median"), numeric_features),
                ]
            )
            model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced", n_jobs=-1))
            ])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            models[target] = model
            metrics[target] = {"accuracy": accuracy, "y_test": y_test, "predictions": predictions, "classes": model.named_steps["classifier"].classes_}
        return models, metrics

    data_source_key = uploaded_file.name if uploaded_file else "default"
    
    with st.spinner("Training Random Forest models... Please wait."):
        models, model_metrics = train_models(df, data_source_key)

    # ==================================================
    # 4. MODEL ACCURACY DASHBOARD
    # ==================================================
    st.header("📊 Model Accuracy Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Interview Accuracy", f"{model_metrics['Interview_Result']['accuracy'] * 100:.2f}%")
    with col2:
        st.metric("Job Search Accuracy", f"{model_metrics['Job_Search_Status']['accuracy'] * 100:.2f}%")
        st.metric("Application Accuracy", f"{model_metrics['Application_Status']['accuracy'] * 100:.2f}%")

    st.divider()

    # ==================================================
    # 5. CONFUSION MATRIX (Static View)
    # ==================================================
    st.header("🎯 Confusion Matrix")
    selected_task = st.selectbox("Select Prediction Task", options=target_columns, format_func=lambda target: target_names[target])
    selected_metrics = model_metrics[selected_task]
    cm = confusion_matrix(selected_metrics["y_test"], selected_metrics["predictions"], labels=selected_metrics["classes"])

    fig_cm = px.imshow(cm, x=selected_metrics["classes"], y=selected_metrics["classes"], text_auto=True, aspect="auto", color_continuous_scale="Blues", labels={"x": "Predicted Result", "y": "Actual Result", "color": "Count"})
    fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Predicted Result", yaxis_title="Actual Result", coloraxis_colorbar=dict(title="Count", thickness=15), dragmode=False)
    st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})

    # ==================================================
    # 6. CANDIDATE PREDICTION FORM
    # ==================================================
    st.header("🤖 Candidate Prediction Form")
    st.write("Enter candidate details to predict recruitment outcomes.")
    education_options = sorted(df["Education"].dropna().astype(str).unique().tolist())
    if not education_options:
        st.error("No valid Education values were found.")
        st.stop()

    with st.form("candidate_form"):
        education = st.selectbox("Education", options=education_options)
        experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=1, step=1)
        skills_score = st.slider("Skills Score", min_value=0, max_value=100, value=70)
        applications_sent = st.number_input("Applications Sent", min_value=0, max_value=1000, value=10)
        coding_score = st.slider("Coding Score", min_value=0, max_value=100, value=70)
        communication_score = st.slider("Communication Score", min_value=0, max_value=100, value=70)
        interview_attended = st.number_input("Interviews Attended", min_value=0, max_value=100, value=1)
        submit = st.form_submit_button("🔮 Predict Outcomes", use_container_width=True)

    # ==================================================
    # 7. PREDICTION RESULTS
    # ==================================================
    if submit:
        candidate = pd.DataFrame([{
            "Education": education, "Experience_Years": experience, "Skills_Score": skills_score,
            "Applications_Sent": applications_sent, "Coding_Score": coding_score,
            "Communication_Score": communication_score, "Interview_Attended": interview_attended,
        }])
        st.divider()
        st.header("✅ Prediction Results")
        
        for target in target_columns:
            model = models[target]
            prediction = model.predict(candidate)[0]
            confidence = float(model.predict_proba(candidate)[0].max() * 100)
            with st.container():
                st.subheader(target_names[target])
                st.success(f"Predicted Result: **{prediction}**")
                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.2f}%")
                st.write("---")

    # ==================================================
    # 8. DATASET PREVIEW & INFO
    # ==================================================
    st.divider()
    with st.expander("📁 View Dataset (Top 20 Rows)"):
        st.dataframe(df.head(20), use_container_width=True)
        
    with st.expander("ℹ️ About HireWise AI"):
        st.write("""
        **Project Name: HireWise AI**
        
        **Algorithm:** Random Forest Classification
        
        **NOTE:** Model is trained on a 70,000 sample subset for performance optimization.
        
        **Author: SARAVANAN.D**""")

except Exception as e:
    st.error("An error occurred while running the app. Please check the details below:")
    st.code(traceback.format_exc())
