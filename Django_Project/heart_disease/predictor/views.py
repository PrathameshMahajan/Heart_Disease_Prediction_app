import os
from django.shortcuts import render
from .forms import HeartDiseaseInputForm
from django.conf import settings
import pandas as pd
import joblib

# Load the Random Forest model
rf_model_path = os.path.join(settings.BASE_DIR, 'random_forest_model.joblib')
if os.path.exists(rf_model_path):
    rf_model = joblib.load(rf_model_path)
else:
    print(f"Error: File '{rf_model_path}' not found.")
    rf_model = None


def predict_heart_disease(request):
    if request.method == 'POST':
        form = HeartDiseaseInputForm(request.POST)
        if form.is_valid():
            input_data = form.cleaned_data

            # Convert input data to DataFrame for prediction
            input_df = pd.DataFrame([input_data])

            # Use the loaded Random Forest model for prediction
            if rf_model is not None:
                prediction_rf = rf_model.predict(input_df)[0]
                if prediction_rf == 1:
                    prediction_message = "You have heart disease."
                else:
                    prediction_message = "You don't have heart disease."
            else:
                print("Error: Random Forest model not available.")
                prediction_message = "No predictions available."

            return render(request, 'result.html', {'prediction_rf': prediction_message})
        else:
            print("Form Errors:", form.errors)  # Debugging line

    return render(request, 'input_form.html', {'form': HeartDiseaseInputForm()})