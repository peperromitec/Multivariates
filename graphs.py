import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt

def plot00(df, ax):
    sns.barplot(
        data=df,
        y=df.index,
        x='edad',
        ax=ax
    )
    ax.set_title("Distribución de edades")
    ax.grid(axis='x', alpha=0.3)
def plot01(df, ax):

