import streamlit as st
from graphs import plot00
import matplotlib.pyplot as plt
from datasetcleaning import upload_dataset
from datasetcleaning import clean_dataset

df = upload_dataset()
dfclean = clean_dataset(df)


st.title("📊 Dashboard")


fig, axs = plt.subplots(2, 2, figsize=(12, 6))

fig.suptitle("Análisis exploratorio")

plot00(df, axs[0, 0])

axs[0, 0].grid()
axs[0, 1].set_title("Gráfica 2")
axs[1, 0].set_title("Gráfica 3")
axs[1, 1].set_title("Gráfica 4")

plt.tight_layout()
plt.show()

st.pyplot(fig)