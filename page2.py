import streamlit as st
from datasetcleaning import upload_dataset
from datasetcleaning import clean_dataset
import streamlit as st




df = upload_dataset()
dfclean = clean_dataset(df)
st.title("📈 Predicciones")