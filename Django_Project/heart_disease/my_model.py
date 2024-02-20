import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_and_save_model(data_path='E:\Prathamesh\Django_Project\Heart Disease data.csv', model_path='random_forest_model.joblib'):
    df = pd.read_csv(data_path)
    df.info()

    # Extract features (X) and target variable (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_rf = rf_model.predict(X_test)

    # Evaluate Random Forest model
    print("\nRandom Forest Classifier:")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))

    # Save the trained model
    joblib.dump(rf_model, model_path)

if __name__ == "__main__":
    train_and_save_model()
