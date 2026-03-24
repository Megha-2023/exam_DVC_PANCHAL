import pandas as pd
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, 'data/processed_data')
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')

def load_data(file_path):
    """Load CSV data from the processed data directory."""
    return pd.read_csv(file_path)

def main():
    """Main function to perform GridSearch for best parameters."""
    # Load the normalized training data
    X_train_scaled = load_data(os.path.join(PROCESSED_DATA_PATH, 'X_train_scaled.csv'))
    y_train = load_data(os.path.join(PROCESSED_DATA_PATH, 'y_train.csv'))

    # Ensure y_train is a DataFrame with one column
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    # Drop the 'date' column if it exists
    if 'date' in X_train_scaled.columns:
        X_train_scaled = X_train_scaled.drop('date', axis=1)
    
    # Define the model and parameters to test
    model = Ridge()
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Save the best parameters to a .pkl file
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(best_params, os.path.join(MODELS_PATH, 'best_params.pkl'))
    print(f"Best parameters saved to {os.path.join(MODELS_PATH, 'best_params.pkl')}")
    print(f"Best parameters: {best_params}")

if __name__ == "__main__":
    main()