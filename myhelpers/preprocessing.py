from sklearn import preprocessing
import pandas as pd
import numpy as np

def typeToObject(d, f):
    return d.astype({f: "object"})
def typeToFloat(d, f):
    return d.astype({f: "float"})
def fillNaWithMeanMode(d):
    for c in d:
        if d[c].dtypes == "object" :
            d[c] = d[c].fillna(value= d[c].mode())
        else:
            d[c] = d[c].fillna(value= d[c].mean())

def input_missing_values(df):
    for col in df.columns:
        if (df[col].dtype is float) or (df[col].dtype is int):
            df[col] = df[col].fillna(df[col].median())
        if (df[col].dtype == object):
            df[col] = df[col].fillna(df[col].mode()[0])
        if (df[col].dtype == np.dtype('<M8[ns]')):
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

def toValidFloat(s):
    return '.'.join(str(s).split(','))

def turnValidFloat(data, col):
    data[col] = data[col].apply(toValidFloat)

def getTitle(str):
    return str.split(',')[1].split('.')[0].strip()

def get_title(name):
    return name.split(',')[1].strip().split()[0]

def addTitles(d):
    d['Title'] = d['Name'].apply(lambda x : getTitle(x))
    
def hasSurname(d):
    d['Surname'] = d['Name'].apply(lambda x : '(' in x)

def isChild(d, age=8):
    d["child"] = d["Age"] < age
    
def isOld(d, age=60):
    d["old"] = d["Age"] > age


def addDummies(d, f):
    X_dummies = pd.get_dummies(d[f], prefix=f, drop_first=False, dummy_na=False, prefix_sep='_')
    return d.join(X_dummies).drop(f, axis=1)



def dummify_features(df):
    """
    Transform categorical variables to dummy variables.

    Parameters
    ----------
    df: dataframe containing only categorical features

    Returns
    -------
    X: new dataframe with dummified features
       Each column name becomes the previous one + the modality of the feature

    enc: the OneHotEncoder that produced X (it's used later in the processing chain)
    """
    colnames = df.columns
    le_dict = {}
    for col in colnames:
        le_dict[col] = preprocessing.LabelEncoder()
        le_dict[col].fit(df[col])
        df.loc[:, col] = le_dict[col].transform(df[col])

    enc = preprocessing.OneHotEncoder()
    enc.fit(df)
    X = enc.transform(df)

    dummy_colnames = [cv + '_' + str(modality) for cv in colnames for modality in le_dict[cv].classes_]
    # for cv in colnames:
    #     for modality in le_dict[cv].classes_:
    #         dummy_colnames.append(cv + '_' + modality)

    return X, dummy_colnames, enc