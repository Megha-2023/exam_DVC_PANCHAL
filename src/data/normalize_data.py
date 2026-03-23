import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, 'data/processed_data')

def load_data(file_path):
    """Load CSV data from the processed data directory."""
    return pd.read_csv(file_path)

def save_data(df, filename):
    """Save DataFrame to the processed data directory."""
    output_path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def normalize_features(X_train, X_test):
    """Normalize the features using StandardScaler."""
    # Exclude the 'date' column if it exists
    if 'date' in X_train.columns:
        date_train = X_train['date']
        date_test = X_test['date']
        X_train_features = X_train.drop('date', axis=1)
        X_test_features = X_test.drop('date', axis=1)
    else:
        X_train_features = X_train
        X_test_features = X_test
        date_train = None
        date_test = None

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit on training data and transform
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_features),
        columns=X_train_features.columns
    )

    # Transform test data
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_features),
        columns=X_test_features.columns
    )

    # Add back the date column if it existed
    if date_train is not None:
        X_train_scaled.insert(0, 'date', date_train.values)
        X_test_scaled.insert(0, 'date', date_test.values)

    return X_train_scaled, X_test_scaled

def main():
    """Main function to normalize the data."""
    # Load the split data
    X_train = load_data(os.path.join(PROCESSED_DATA_PATH, 'X_train.csv'))
    X_test = load_data(os.path.join(PROCESSED_DATA_PATH, 'X_test.csv'))

    # Normalize the features
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    # Save the normalized data
    save_data(X_train_scaled, 'X_train_scaled.csv')
    save_data(X_test_scaled, 'X_test_scaled.csv')

if __name__ == "__main__":
    main()