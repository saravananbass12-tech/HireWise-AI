import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.title("Test App")
st.success("Imports are working perfectly!")

st.write("pandas version:", pd.__version__)
st.write("plotly version:", px.__version__)
st.write("scikit-learn version is working.")
