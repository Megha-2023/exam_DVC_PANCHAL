import pandas as pd
import joblib
import pickle
import os
from sklearn import ensemble

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, 'data/processed_data')
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')

def load_data(file_path):
    """ Load CSV data from the processed data """
    return pd.read_csv(file_path)

def main():
    """ Main function to train model as per best parameters """
    # Load normalized training data
    X_train_scaled = load_data(os.path.join(PROCESSED_DATA_PATH, "X_train_scaled.csv"))    
    y_train = load_data(os.path.join(PROCESSED_DATA_PATH, "y_train.csv"))    

    # Ensure y_train is a DataFrame with one column
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    # Drop the 'date' column if it exists
    if 'date' in X_train_scaled.columns:
        X_train_scaled = X_train_scaled.drop('date', axis=1)

    # Load best parameters stored .pkl file
    params_file = os.path.join(MODELS_PATH, 'best_params.pkl')
    with open(params_file, 'rb') as f:
        best_params = pickle.load(f)
    
    # Filter best params to include keys that RF accepts
    valid_keys = ensemble.RandomForestRegressor().get_params().keys()

    filtered_params = {k: v for k, v in best_params.items() if k in valid_keys}

    # Initialize and train model with filetered parameters
    print("Training Model.......")
    model = ensemble.RandomForestRegressor(**filtered_params)
    model.fit(X_train_scaled, y_train)

    # Save the trained model to file
    model_file = os.path.join(MODELS_PATH, 'trained_model.joblib')
    joblib.dump(model, model_file)
    print("Model trained and saved Successfully !")


if __name__ == "__main__":
    main()
