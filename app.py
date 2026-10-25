import streamlit as st
import pandas as pd
import plotly.express as px
import traceback

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

# மொபைலில் வடிவமைப்பை அழகாக்க தனிப்பட்ட CSS
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

try:
    # ==================================================
    # 1. LOAD DATASET FROM HUGGING FACE
    # ==================================================
    HF_CSV_URL = "https://huggingface.co/datasets/saroa/hirewise-dataset/resolve/main/human_job_prediction_50000_records_converted.csv"

    @st.cache_data
    def load_data(url):
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
            "Application_Status": [
                "application_status", "job_selection_status", "selection_status",
                "selection_result", "job_status", "hiring_status", "final_status",
                "job_outcome", "recruitment_status", "application_result"
            ],
        }

        rename_map = {}
        for standard_name, possible_names in aliases.items():
            if standard_name in data.columns:
                continue
            for possible_name in possible_names:
                actual_name = normalized.get(possible_name)
                if actual_name is not None:
                    rename_map[actual_name] = standard_name
                    break

        if rename_map:
            data = data.rename(columns=rename_map)

        required_columns = [
            "Education", "Experience_Years", "Skills_Score", "Applications_Sent",
            "Coding_Score", "Communication_Score", "Interview_Attended",
            "Job_Search_Status", "Interview_Result", "Application_Status"
        ]
        missing_columns = [column for column in required_columns if column not in data.columns]

        if missing_columns:
            available_columns = ", ".join(map(str, data.columns.tolist()))
            raise ValueError(
                "Some required columns could not be matched: " + ", ".join(missing_columns)
                + ". Available CSV columns are: " + available_columns
            )

        data["Application_Status"] = (
            data["Application_Status"].astype("string").str.strip().replace({
                "Selected": "Approved", "selected": "Approved", "Approved": "Approved",
                "Not Selected": "Rejected", "Not selected": "Rejected",
                "Rejected": "Rejected", "Rejected Application": "Rejected"
            })
        )

        for column in ["Job_Search_Status", "Interview_Result", "Application_Status"]:
            data[column] = data[column].astype("string").str.strip()
            data[column] = data[column].replace(["", "nan", "None", "<NA>"], pd.NA)

        for column in [
            "Experience_Years", "Skills_Score", "Applications_Sent",
            "Coding_Score", "Communication_Score", "Interview_Attended"
        ]:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        data["Education"] = data["Education"].astype("string").str.strip()
        data["Education"] = data["Education"].replace(["", "nan", "None", "<NA>"], pd.NA)
        return data

    with st.spinner("Hugging Face-ல் இருந்து Dataset Load ஆகிறது... தயவுசெய்து காத்திருங்கள்."):
        df = load_data(HF_CSV_URL)

    if df.empty:
        st.error("The dataset is empty. Please check the CSV file.")
        st.stop()

    st.success(f"Dataset loaded successfully: {len(df):,} records")

    # ==================================================
    # 2. INPUT AND OUTPUT COLUMNS
    # ==================================================
    feature_columns = [
        "Education", "Experience_Years", "Skills_Score", "Applications_Sent",
        "Coding_Score", "Communication_Score", "Interview_Attended",
    ]

    target_columns = [
        "Job_Search_Status", "Interview_Result", "Application_Status",
    ]

    target_names = {
        "Job_Search_Status": "Job Search Status",
        "Interview_Result": "Interview Result",
        "Application_Status": "Application Status",
    }

    categorical_features = ["Education"]
    numeric_features = [
        "Experience_Years", "Skills_Score", "Applications_Sent",
        "Coding_Score", "Communication_Score", "Interview_Attended",
    ]

    # ==================================================
    # 3. TRAIN RANDOM FOREST MODELS (Using Full Data safely)
    # ==================================================
    @st.cache_resource
    def train_models(data):
        models = {}
        metrics = {}

        for target in target_columns:
            # Dataset-ல் உள்ள முழு வரிகளையும் (100%) எடுக்கிறோம்
            model_data = data[feature_columns + [target]].dropna(subset=[target]).copy()
            model_data = model_data[model_data[target].astype("string").str.strip().ne("")]

            if model_data.empty:
                raise ValueError(f"No valid target values were found for {target}.")

            if model_data[target].nunique() < 2:
                raise ValueError(f"{target} must contain at least two different classes.")

            class_counts = model_data[target].value_counts()
            if class_counts.min() < 2:
                raise ValueError(f"Every class in {target} needs at least two records.")

            X = model_data[feature_columns]
            y = model_data[target].astype(str)

            # 80% Training, 20% Testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y,
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("education",
                        Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]), categorical_features),
                    ("numbers", SimpleImputer(strategy="median"), numeric_features),
                ]
            )

            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(
                        n_estimators=50, 
                        random_state=42, 
                        class_weight="balanced", 
                        n_jobs=-1,
                        max_samples=0.1  # ⚠️ மெமரி க்ராஷ் ஆகாமல் இருக்க, ஒவ்வொரு மரமும் 10% தரவை மட்டும் எடுக்கும்
                    )),
                ]
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            models[target] = model
            metrics[target] = {
                "accuracy": accuracy,
                "y_test": y_test,
                "predictions": predictions,
                "classes": model.named_steps["classifier"].classes_,
            }

        return models, metrics

    with st.spinner("Training Random Forest models... Please wait (This may take a minute)."):
        models, model_metrics = train_models(df)

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

    selected_task = st.selectbox(
        "Select Prediction Task",
        options=target_columns,
        format_func=lambda target: target_names[target],
    )

    selected_metrics = model_metrics[selected_task]
    y_test = selected_metrics["y_test"]
    predictions = selected_metrics["predictions"]
    classes = selected_metrics["classes"]

    cm = confusion_matrix(y_test, predictions, labels=classes)

    fig_cm = px.imshow(
        cm, x=classes, y=classes, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "Predicted Result", "y": "Actual Result", "color": "Count"},
    )
    
    fig_cm.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Predicted Result",
        yaxis_title="Actual Result",
        coloraxis_colorbar=dict(title="Count", thickness=15),
        dragmode=False, 
    )
    
    st.plotly_chart(
        fig_cm, 
        use_container_width=True, 
        config={'displayModeBar': False, 'staticPlot': True} 
    )

    # ==================================================
    # 6. CANDIDATE PREDICTION FORM
    # ==================================================
    st.header("🤖 Candidate Prediction Form")
    st.write("Enter candidate details to predict recruitment outcomes.")

    education_options = sorted(df["Education"].dropna().astype(str).unique().tolist())
    if not education_options:
        st.error("No valid Education values were found in the dataset.")
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
            "Education": education,
            "Experience_Years": experience,
            "Skills_Score": skills_score,
            "Applications_Sent": applications_sent,
            "Coding_Score": coding_score,
            "Communication_Score": communication_score,
            "Interview_Attended": interview_attended,
        }])

        st.divider()
        st.header("✅ Prediction Results")
        
        for target in target_columns:
            model = models[target]
            prediction = model.predict(candidate)[0]
            probabilities = model.predict_proba(candidate)[0]
            confidence = float(probabilities.max() * 100)

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
        **Project Name:** HireWise AI
        **Algorithm:** Random Forest Classification
        **Note:** Model is trained on the full dataset records using an optimized algorithm to prevent memory crashes.
        **Author: SARAVANAN.D**
        """)

except Exception as e:
    st.error("An error occurred while running the app. Please check the details below:")
    st.code(traceback.format_exc())
