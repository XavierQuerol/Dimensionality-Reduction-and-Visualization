import sys

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

import pandas as pd

### PREPROCESSING

def drop_columns(df, column_names):
    df = df.drop(columns = column_names)
    return df

def drop_rows_sick(df, column_names):
    df = df.dropna(subset=column_names)
    df = df[df["age"] <= 120] # Drop ages over 120
    df = df.reset_index(drop=True)

    return df

"""
Applies a minmaxscaler to all numerical columns.
If it finds a nan in a numerical column it removes the instance.
"""
def min_max_scaler(df, numerical_cols=slice(None)):

    scaler = MinMaxScaler()

    # Scaler Training with all the train and test information.
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def one_hot_encoding(df):
    categorical_features = df.select_dtypes(include=['object']).nunique()[lambda x: x > 2].index.tolist()

    ohe = OneHotEncoder(handle_unknown='ignore')

    encoded_array = ohe.fit_transform(df[categorical_features]).toarray()

    # Create new column names for the encoded features
    new_cols = [f'{col}_{cat}' for col in categorical_features for cat in
                ohe.categories_[categorical_features.index(col)]]

    # Create a DataFrame for the encoded features
    df_encoded = pd.DataFrame(encoded_array, columns=new_cols, index=df.index)

    # Substitute the original categorical features with the new numeric ones
    df = df.drop(categorical_features, axis=1)
    df = df.join(df_encoded)

    return df

def binary_encoding(df):

    binary_features = df.select_dtypes(include=['object']).nunique()[lambda x: x <= 2].index.tolist()

    # Encode only the binary features
    for feature in binary_features:

        label_encoder = LabelEncoder()
        df[feature] = label_encoder.fit_transform(df[feature])

    return df

def label_encoder(df):
    le = LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])
    return df



def fill_nans(df, columns_predict):

    model = LinearRegression()

    # Train with all columns except the ones to predict
    cols = [col for col in df.columns if col not in columns_predict]

    for col in columns_predict:
        df_model = df.dropna(subset=[col])
        df_nans = df[df[col].isna()]

        if not df_model.empty:
            x = df_model[cols]
            y = df_model[col]

            model.fit(x, y)

            if not df_nans.empty:
                df.loc[df_nans.index, col] = model.predict(df_nans[cols])

    return df

#### UTILS REGARDING INPUT

def get_user_choice(prompt, options, is_numeric = False, is_float = False):
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            if is_numeric:
                print(f"  {option}")
            else:
                print(f" {i}. {option}")
        choice = input("Please enter the number of your choice: ")

        if is_numeric and is_float and float(choice) in options:
            return float(choice)
        if not is_float and is_numeric and int(choice) in options:
            return int(choice)
        if choice in options:
            return choice
        if not is_numeric and choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Invalid choice. Try again.\n")

def loading_bar(iteration, total, length=40):
    percent = (iteration / total)
    bar_length = int(length * percent)
    bar = '#' * bar_length + '-' * (length - bar_length)
    sys.stdout.write(f'\r[{bar}] {percent:.2%} Complete')
    sys.stdout.flush()