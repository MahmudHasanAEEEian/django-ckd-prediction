# myapp/views.py

from django.shortcuts import render
import joblib
import pandas as pd
from .forms import PredictionForm

# Correct model paths relative to the project root
models = {
    'knn': "mlscripts/models/knn_model.pkl",
    'svm': "mlscripts/models/svm_model.pkl",
    'gnb': "mlscripts/models/gnb_model.pkl",
    'lr': "mlscripts/models/lr_model.pkl",
    'sgd': "mlscripts/models/sgd_model.pkl",
    'stacking': "mlscripts/models/stacking_model.pkl",
}

# Load models
loaded_models = {key: joblib.load(value) for key, value in models.items()}

def map_choice_to_numeric(value, choice_map):
    return choice_map.get(value, 0)  # default to 0 if the value is not found

def predict_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract form data and map categorical values to numerical values
            form_data = form.cleaned_data
            categorical_mapping = {
                'rbc': {'normal': 1, 'abnormal': 0},
                'pc': {'normal': 1, 'abnormal': 0},
                'pcc': {'present': 1, 'notpresent': 0},
                'ba': {'present': 1, 'notpresent': 0},
                'htn': {'yes': 1, 'no': 0},
                'dm': {'yes': 1, 'no': 0},
                'cad': {'yes': 1, 'no': 0},
                'appet': {'good': 1, 'poor': 0},
                'pe': {'yes': 1, 'no': 0},
                'ane': {'yes': 1, 'no': 0},
            }

            # Convert categorical fields
            for key, value in categorical_mapping.items():
                form_data[key] = map_choice_to_numeric(form_data[key], value)

            # Arrange features and ensure the model's expected input order
            features = [form_data[field] for field in form.fields]
            sample_df = pd.DataFrame([features], columns=[field for field in form.fields])

            # Get predictions from all models
            results = {}
            for model_name, model in loaded_models.items():
                prediction = model.predict(sample_df)
                results[model_name] = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

            return render(request, 'result.html', {'results': results})
    else:
        form = PredictionForm()

    return render(request, 'index.html', {'form': form})
