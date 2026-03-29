from data_analyzer import data_loader, data_analyzer
from sklearn.preprocessing import OneHotEncoder , MinMaxScaler
import pandas as pd



def drop_columns(df, profile):
    drop_col = profile["analysis"]["preprocessing"]["drop_cols"]
    df1 = df.drop(columns = drop_col, errors = "ignore")
    return df1

def fill_numeric(df, profile):
    num_cols = profile["analysis"]["preprocessing"]["fill_mean"]
    for col in num_cols:
        df[col] =  df[col].fillna(df[col].mean())
    return df

def fill_categorical(df, profile):
    cat_cols = profile["analysis"]["preprocessing"]["fill_mode"]
    for col in cat_cols:
        df[col] =  df[col].fillna(df[col].mode().iat[0])
    return df

def encode_ohe(df, profile):
    encode_cols = profile["analysis"]["preprocessing"]["encode"]
    
    for col in encode_cols:
        if df[col].nunique() <= 3:
            ohe = pd.get_dummies(df[col], prefix = col)
            df = pd.concat([df, ohe], axis=1)
            df = df.drop(col, axis = 1)
    return df

def scaling(df, profile):
    scale_col = profile["analysis"]["preprocessing"]["scale"]
    scaler = MinMaxScaler()

    for col in scale_col:
        if col not in profile["analysis"]["preprocessing"]["drop_cols"]:
            df[col] = scaler.fit_transform(df[[col]])
    return df
   
    

def preprocess_pipeline(df, profile):

    trf1 = drop_columns(df,profile)
    trf2 = fill_numeric(trf1, profile)
    trf3 = fill_categorical(trf2, profile)
    trf4 = encode_ohe(trf3, profile)
    df = scaling(trf4, profile)

    return df

if __name__ == "__main__":
    preprocess_pipeline()




