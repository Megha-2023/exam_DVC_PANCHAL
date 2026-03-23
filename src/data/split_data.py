import pandas as pd
from sklearn.model_selection import train_test_split
import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_PATH, 'data/raw_data/raw.csv')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'data/processed_data')

print(PROJECT_PATH)
print(DATA_PATH)


def check_existing_file(file_path):
    """Check if a file already exists. If it does, ask if we want to overwrite it"""
    if os.path.isfile(file_path):
        while True:
            response = input(f"File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True

def main():
    """Read data, Split and save to processed folder"""

    # Read raw CSV file
    df = pd.read_csv(DATA_PATH)
    target = df['silica_concentrate']
    features = df.drop(['silica_concentrate'], axis=1)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=42
    )

    # Save dataframes to processed folder
    for file, filename in zip(
        [X_train, X_test, y_train, y_test],
        ['X_train', 'X_test', 'y_train', 'y_test']
        ):
        output_file = os.path.join(OUTPUT_PATH, f'{filename}.csv')
        if check_existing_file(output_file):
            file.to_csv(output_file, index=False)
    

if __name__ == "__main__":
    main()
