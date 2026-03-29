import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

def data_loader(path):
    df = pd.read_csv(path)

    return df

def data_analyzer(df):

    data_profile = {
        "basic": {
            'Rows' : df.shape[0],
            'Columns' : df.shape[1],
            'Duplicates': df.duplicated().sum(),
            },
        "quality": {
            'n_cols_null' : df.isnull().sum().to_dict(),
            'total_null'   : df.isnull().sum().sum(),
            'duplicate_count' : df.duplicated().sum(),
            'null_percent' : {}
        },
        "columns" : {
            "numeric" :  [],
            "categorical" : [],
            "unique_counts" : {}
        },
    
        "analysis": {
            "identifier_cols" : [],
            "constant_cols" : [],
            "low_cardinality_numeric" : [],
            "high_cardinality_categorical" : [],
            "preprocessing" : {
                "drop_cols" : [],
                "fill_mean" : [],
                "fill_mode" : [],
                "encode" : [],
                "scale" : []
            }
        }

    }

    for col, value in data_profile["quality"]["n_cols_null"].items():
        if value != 0:
            data_profile["quality"]["null_percent"][col] = round( value / df.shape[0] * 100 , 2 )

    num_cols = []
    cat_cols = []

    for col in df.columns:
        data_profile["columns"]["unique_counts"][col] = df[col].nunique()

        if is_numeric_dtype(df[col]):
            num_cols.append(col)

        else :
            cat_cols.append(col)

    data_profile["columns"]["numeric"] = num_cols
    data_profile["columns"]["categorical"] = cat_cols

    identifier = []
    constant = []
    low_cardinality = []
    high_cardinality = []
    
    for key, value in data_profile["columns"]["unique_counts"].items():
        #identifiers
        if value == df.shape[0]:
            identifier.append(key)

        #costant    
        elif value == 1:
            constant.append(key)

        #low cardinality numeric
        elif key in data_profile['columns']["numeric"] and value < 10:
            low_cardinality.append(key)

        #high cardinality categorical
        elif key in data_profile["columns"]["categorical"] and value > 50:
            high_cardinality.append(key)

    data_profile["analysis"]["identifier_cols"] = identifier
    data_profile["analysis"]["constant_cols"] = constant
    data_profile["analysis"]["low_cardinality_numeric"] = low_cardinality
    data_profile["analysis"]["high_cardinality_categorical"] = high_cardinality

    drop_cols = data_profile["analysis"]["identifier_cols"].copy()

    for key, value in data_profile["quality"]["null_percent"].items():
        if value > 55:
            drop_cols.append(key) 

    data_profile["analysis"]["preprocessing"]["drop_cols"] = drop_cols

    fill_mean = []
    fill_mode = []

    for key,value in data_profile["quality"]["null_percent"].items():
        if key in data_profile["columns"]["numeric"] and df[key].isnull().sum() > 0:
            if key in data_profile["analysis"]["preprocessing"]["drop_cols"]:
                pass
            else:
                fill_mean.append(key)

        elif key in data_profile["columns"]["categorical"] and df[key].isnull().sum() > 0:
            if key in data_profile["analysis"]["preprocessing"]["drop_cols"]:
                pass
            else:
                fill_mode.append(key)
        
    data_profile["analysis"]["preprocessing"]["fill_mean"] = fill_mean
    data_profile["analysis"]["preprocessing"]["fill_mode"] = fill_mode

    encode = []

    for key in data_profile["columns"]["categorical"]:
        if key not in data_profile["analysis"]["preprocessing"]["drop_cols"]:
            if not is_numeric_dtype(df[key]):
                encode.append(key)
    data_profile["analysis"]["preprocessing"]["encode"] = encode

    scale = []

    for key, value in data_profile["columns"]["unique_counts"].items():
        if key in data_profile["columns"]["numeric"] and value > 10:
            scale.append(key)
    data_profile["analysis"]["preprocessing"]["scale"] = scale
      

    return data_profile

def pipeline(path):
    df = data_loader(path)
    return df, data_analyzer(df)


if __name__ == "__main__":
    pipeline()
