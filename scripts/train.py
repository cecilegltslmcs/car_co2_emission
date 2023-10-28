# libraries used for data exploration
import numpy as np
import pandas as pd

# librairies used for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# utils
import pickle

PATH = "data/dataset.csv"
OUTPUT_FILE = "random_forest.bin"

def prepare_data(PATH):
    df = pd.read_csv(PATH)
    df = df[df["energy"] != 'electric']

    categorical = list(df.dtypes[df.dtypes == 'object'].index)
    numerical = [i for i in list(df.columns) if i not in categorical]
    numerical.pop(-1)
    df["electrical_autonomy"] = df["electrical_autonomy"].fillna(0)
    df["electrical_nominal_power"] = df["electrical_nominal_power"].fillna(0)
    df["elect_conso"] = df["elect_conso"].fillna(0)
    df["city_electrical_autonomy"] = df["city_electrical_autonomy"].fillna(0)
    for c in numerical:
        df[c] = df[c].fillna(df[c].median())
    df["co2_emission"] = df["co2_emission"].fillna(df["co2_emission"].median())
    df["co2_emission"] = np.log1p(df["co2_emission"])

    del df["electrical_nominal_power"]
    del df["elect_conso"]
    del df["electrical_autonomy"]
    del df["city_electrical_autonomy"]
    
    return df

def preprocessing_data(df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train["co2_emission"].values
    y_test = df_test["co2_emission"].values

    del df_train['co2_emission']
    del df_test['co2_emission']

    return df_train, df_test, y_train, y_test

def train(df, y_train):
    train_dicts = df.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    rf = RandomForestRegressor(
        max_depth=10,
        n_estimators=120,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    return dv, rf


def predict(dv, model, df):
    test_dicts = df.to_dict(orient='records')
    X_test = dv.transform(test_dicts)
    y_pred = model.predict(X_test)
    
    return y_pred


data = prepare_data(PATH)
df_train, df_test, y_train, y_test = preprocessing_data(data)
dv, model = train(df_train, y_train)
y_pred = predict(dv, model, df_test)

rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 5)
print(f"rmse: {rmse}")

with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {OUTPUT_FILE}")