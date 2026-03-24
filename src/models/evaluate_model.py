import pandas as pd
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, 'data/processed_data')
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')
METRICS_PATH = os.path.join(PROJECT_PATH, 'metrics')


def load_data(file_path):
    """ Load CSV data from the processed data """
    return pd.read_csv(file_path)


def main():
    """ Main function to evaluate the trained model """

    # Load normalized test data
    X_test_scaled = load_data(os.path.join(PROCESSED_DATA_PATH, 'X_test_scaled.csv'))
    y_test = load_data(os.path.join(PROCESSED_DATA_PATH, 'y_test.csv'))

    # Ensure y_train is a DataFrame with one column
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    # Drop the 'date' column if it exists
    if 'date' in X_test_scaled.columns:
        X_test_scaled = X_test_scaled.drop('date', axis=1)
    
    # Load Model 
    model_file = os.path.join(MODELS_PATH, 'trained_model.joblib')
    model = joblib.load(model_file)

    # Generate predictions
    print("Predicting silica_concentrate....")
    predictions = model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # New data containing predictions
    results_df = pd.DataFrame(
        {'Actual': y_test,
         'Predicted': predictions}
    )
    result_file = os.path.join(PROCESSED_DATA_PATH, 'silica_predictions.csv')
    results_df.to_csv(result_file, index=False)

    # Save score to metrics folder
    scores = {
        "mean_squared_error": float(mse),
        "r2_score": float(r2)
    }
    os.makedirs(METRICS_PATH, exist_ok=True)
    with open(os.path.join(METRICS_PATH, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)
    print("Metrics saved Successfully to scores.json file !")


if __name__ == "__main__":
    main()
