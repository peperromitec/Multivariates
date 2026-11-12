import pandas as pd
def upload_dataset():
    df = pd.read_excel("Multivariates\dataset_dummy_tarjetas_santander_10000_clientes.csv (1).xlsx")
    return df

def clean_dataset(df):
    # Remove duplicates
    df = df.dropna()
    
    return df
