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

def clean_and_process_dataset(PATH:str) -> pd.DataFrame:
    """
    Load a dataset from the given file path and return a cleaned version.

    Args:
        PATH (str): The path to the dataset file.

    Returns:
        DataFrame: A Pandas DataFrame containing the cleaned dataset.
    """
    df = pd.read_csv(PATH)

    # Remove rows with 'electric' in the 'energy' column
    df = df[df["energy"] != 'electric']

    # Define lists of categorical and numerical columns
    categorical = list(df.select_dtypes(include=['object']).columns)
    numerical = list(df.select_dtypes(exclude=['object']).columns)
    numerical.pop(-1)

    # Fill missing values in specific columns
    columns_to_fill = [
        "electrical_autonomy", 
        "electrical_nominal_power", 
        "elect_conso", 
        "city_electrical_autonomy"
        ]
    for col in columns_to_fill:
        df[col] = df[col].fillna(0)
    
    # Fill missing values in numerical columns with their medians
    df[numerical] = df[numerical].fillna(df[numerical].median())


    # Log-transform the 'co2_emission' column
    df["co2_emission"] = np.log1p(df["co2_emission"])

    # Remove unwanted columns
    columns_to_remove = [
        "electrical_nominal_power", 
        "elect_conso", 
        "electrical_autonomy", 
        "city_electrical_autonomy"
        ]
    df = df.drop(columns=columns_to_remove, axis=1)
    
    return df

def split_and_prepare_data(df:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, np.array, np.array):
    """
    Splits the dataset into training and testing sets and prepares them for modeling.

    Args:
        df (pd.DataFrame): The dataset to split.

    Returns:
        pd.DataFrame: Training data without the 'co2_emission' column.
        pd.DataFrame: Testing data without the 'co2_emission' column.
        np.array: Target values for the training data.
        np.array: Target values for the testing data.
    """
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train["co2_emission"].values
    y_test = df_test["co2_emission"].values

    del df_train['co2_emission']
    del df_test['co2_emission']

    return df_train, df_test, y_train, y_test

def train_random_forest_model(df:pd.DataFrame, y_train:np.array) -> (DictVectorizer, RandomForestRegressor):
    """
    Trains a Random Forest regression model and returns a DictVectorizer and the trained model.

    Args:
        df (pd.DataFrame): Dataset with the training data.
        y_train (np.array): Array with the target variable.

    Returns:
        Tuple[DictVectorizer, RandomForestRegressor]: A tuple containing a DictVectorizer and a trained Random Forest model.
    """
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


def predict_co2_emission(dv: DictVectorizer, model: RandomForestRegressor, df: pd.DataFrame) -> np.array:
    """
    Predicts CO2 emissions of cars and returns the predicted values.

    Args:
        dv (DictVectorizer): DictVectorizer for transforming car information.
        model (RandomForestRegressor): Trained model for predicting CO2 emissions.
        df (pd.DataFrame): Information of cars in a DataFrame.

    Returns:
        np.array: An array containing the predicted CO2 emissions.
    """
    test_dicts = df.to_dict(orient='records')
    X_test = dv.transform(test_dicts)
    y_pred = model.predict(X_test)
    
    return y_pred


data = clean_and_process_dataset(PATH)
df_train, df_test, y_train, y_test = split_and_prepare_data(data)
dv, model = train_random_forest_model(df_train, y_train)
y_pred = predict_co2_emission(dv, model, df_test)

rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 5)
print(f"rmse: {rmse}")

with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {OUTPUT_FILE}")