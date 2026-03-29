
import pandas as pd
from preprocessing import preprocess_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pandas.api.types import is_numeric_dtype
from tabulate import tabulate

from data_analyzer import data_loader, data_analyzer

def identify_target(df):
    target = ""
    while target not in df.columns:
        print("which of the following is the target column: \n", list(df.columns) )
        target = input()
    return target

def separate(df, target):
    X = df.drop(target, axis = 1)
    y = df[target]
    delete = []
    for col in X.columns:
        if not is_numeric_dtype(X[col]):
            delete.append(col)
    X = X.drop(delete , axis = 1)

    return X, y

# def split(df):
    
#     target = identify_target(df)
#     X, y = separate(df, target)

#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state= 42)

#     return X_train,X_test,y_train,y_test

def problem_type_detector(y):
    problem_type = None
    model_suggested = []
    accuracy_metric = None
    if is_numeric_dtype(y) and y.nunique() > 15:
        problem_type = "Regression"
        model_suggested = [("Linear Regression" , Linear_Regression_trainer) , ("Random Forest", Random_Forest_trainer)]
        scoring_metric = "r2"
        folds = 5
    else:
        problem_type = "Classification"
        model_suggested = [("Logistic Regression" , Logistic_Regression_trainer) , ("Random Forest", Random_Forest_trainer) ,("KNearest Neighbour", KNN_trainer)]
        scoring_metric = "accuracy"      
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return problem_type, model_suggested, scoring_metric, folds

def Logistic_Regression_trainer():
    lr = LogisticRegression()
        
    return lr

def Linear_Regression_trainer():
    lir = LinearRegression()
        
    return lir

def Random_Forest_trainer():
    rf = RandomForestClassifier()
    
    return rf

def KNN_trainer():
    knn = KNeighborsClassifier()
    
    return knn

# def Accuracy_score(y_test, y_pred):
#     return accuracy_score(y_test, y_pred)

# def R2_score(y_test, y_pred):
#     return r2_score(y_test, y_pred)

def model_trainer(problem_type, model_suggested, X, y, scoring_metric, folds ):
    result = {}
    
    for name, mod in model_suggested:
        model = mod()
        score = cross_val_score(model ,X, y, cv=folds, scoring = scoring_metric)
        result[name] = score.mean()

    return result            

def Model_comparison(result):    

    best_performance_dict = dict(sorted(result.items(), key = lambda x: x[1],  reverse = True))

    table = [(model, score) for model, score in best_performance_dict.items()]

    print(tabulate(table, headers = ["model" , "Cross Val Score"], tablefmt = "fancy_grid"))
    
    best_model = next(iter(best_performance_dict))
    score = next(iter(best_performance_dict.values()))
    print("Best model = ", best_model)
    print("Cross Validation Score = ", score)

def run_pipeline():
    df = data_loader("train.csv").copy()
    profile = data_analyzer(df)
    df_clean = preprocess_pipeline(df, profile)
    # X_train,X_test,y_train,y_test = split(df_clean)
    target = identify_target(df_clean)
    X, y = separate(df_clean, target)
    problem_type, model_suggested, scoring_metric, folds = problem_type_detector(y)
    result = model_trainer(problem_type, model_suggested, X, y, scoring_metric, folds)
    Model_comparison(result)
    print("Target Column = " , target)
    print("Problem Type = " , problem_type)
    print("Models Tested = " , len(model_suggested))
    print("Scoring Metric = " , scoring_metric)

if __name__ == "__main__":
    run_pipeline()

