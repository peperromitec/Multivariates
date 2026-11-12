import streamlit as st

dashboard = st.Page(
    "page1.py",
    title="Dashboard",
    icon="📊",
    default=True
)

analisis = st.Page(
    "page2.py",
    title="Machine Learning",
    icon="📈"
)

pg = st.navigation([dashboard, analisis])

pg.run()