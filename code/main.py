import os
import pandas as pd
from preprocessing import preprocess_vowel, preprocess_sick

def preprocess_datasets():
    df_sick_X, df_sick_y = preprocess_sick()
    df_vowel_X, df_vowel_y = preprocess_vowel()

def load_ds(name):

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_processed")
    file = os.path.join(base_dir,f'{name}.csv')
    df = pd.read_csv(file)

    # We remove the class of the dataset as we will not be using it
    return df.iloc[:,:-1], df.iloc[:,-1]


def  main():
    print("Dimensionality")

if __name__ == "__main__":
    main()