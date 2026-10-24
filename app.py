import streamlit as st
import pandas as pd

st.title("Test App")
st.write("App is working perfectly!")

try:
    df = pd.read_csv("https://huggingface.co/datasets/saroa/hirewise-dataset/resolve/main/human_job_prediction_50000_records_converted.csv")
    st.success(f"Dataset loaded successfully! Total rows: {len(df)}")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Dataset Error: {e}")
