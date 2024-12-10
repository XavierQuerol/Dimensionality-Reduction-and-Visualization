from scipy.io.arff import loadarff
import pandas as pd

from code.utils import drop_rows_sick, drop_columns, min_max_scaler, one_hot_encoding, label_encoder, binary_encoding, fill_nans

def get_dataset(dataset):
    if dataset == "sick":
        return preprocess_sick()
    elif dataset == "vowel":
        return preprocess_vowel()
    else:
        raise ValueError("This dataset is not supported")

def preprocess_sick():
    """
    Applies the specified preprocessings for the dataset sick and stores it in the file datasets_processed/sick.csv.
    :return: dataframe
    """
    # Load arff file
    df_sick, meta_train = loadarff(f'datasets/sick.arff')

    # Define datasets
    df_sick = pd.DataFrame(df_sick)

    # Decode utf8 columns
    for col in df_sick.columns:
        df_sick[col].map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Pick up the columns that have nans
    columns_with_nans = []
    for col in df_sick.columns:
        sum_nans = sum(df_sick[col].isna())
        percentage = sum_nans / len(df_sick) * 100
        if (percentage > 0.1) and (percentage < 99):
            columns_with_nans.append(col)

    # We remove the rows where the age was null and where it is greater than 100, which means there was an error (there are two)
    df_sick = drop_rows_sick(df_sick, ['age'])

    df_sick = drop_columns(df_sick, ['TBG_measured', 'TBG'])
    df_sick = min_max_scaler(df_sick, ['age'])
    df_sick = one_hot_encoding(df_sick)
    df_sick = binary_encoding(df_sick)
    df_sick = fill_nans(df_sick, columns_with_nans)
    df_sick = min_max_scaler(df_sick, columns_with_nans)

    df_sick = df_sick[[col for col in df_sick if col != 'sick'] + ['sick']]

    df_sick.to_csv(f'datasets_processed/sick.csv', index=False)

    return df_sick.iloc[:,:-1], df_sick.iloc[:,-1]

def preprocess_vowel():
    """
    Applies the specified preprocessings for the dataset vowel and stores it in the file datasets_processed/vowel.csv.
    :return: dataframe
    """
    # Load arff file
    df_vowel, meta_train = loadarff(f'datasets/vowel.arff')

    # Define datasets
    df_vowel = pd.DataFrame(df_vowel)

    y = df_vowel[['Class']]
    df_vowel = df_vowel.drop(columns=['Class','Train_or_Test'])

    df_vowel = one_hot_encoding(df_vowel)
    df_vowel = binary_encoding(df_vowel)
    df_vowel = min_max_scaler(df_vowel)
    y = label_encoder(y)

    df_vowel = pd.concat([df_vowel, y], axis=1)

    df_vowel.to_csv(f'datasets_processed/vowel.csv', index=False)

    return df_vowel.iloc[:,:-1], df_vowel.iloc[:,-1]


