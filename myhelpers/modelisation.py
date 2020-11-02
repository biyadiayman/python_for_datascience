from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
    
def get_model_RegLog(X, y):
    """
    Takes X and target: y and returns a Logistic Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    
    return {"y_test": y_test,
            "y_pred": y_pred,
            "score_train": score_train,
            "score_test": score_test,
            "model": model,
           "report": report}
               
def get_model_RegLin(X, y):
    """
    Takes X and target: y and returns a Logistic Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    score_test = model.score(X_test, y_test)
    
    return {"score_test": score_test,
            "model": model}


def rndForrestModel(X, y):
    """
    Takes X and target: y and returns a Random Forrest model
    """    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42, max_depth=20)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    
    return {"y_test": y_test,
            "y_pred": y_pred,
            "score_train": score_train,
            "score_test": score_test,
            "model": model,
           "report": report}

def parse_model(X, use_columns):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    X = X[use_columns]
    return X, target