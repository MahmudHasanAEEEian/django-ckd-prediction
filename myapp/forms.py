# mlmodels/forms.py

from django import forms

class PredictionForm(forms.Form):
    bgr = forms.FloatField(label='Blood Glucose Random')
    bu = forms.FloatField(label='Blood Urea')
    sc = forms.FloatField(label='Serum Creatinine')
    sod = forms.FloatField(label='Sodium')
    hemo = forms.FloatField(label='Hemoglobin')
    pcv = forms.FloatField(label='Packed Cell Volume')
    rc = forms.FloatField(label='Red Blood Cell Count')
    sg = forms.FloatField(label='Specific Gravity')
    al = forms.FloatField(label='Albumin')
    su = forms.FloatField(label='Sugar')
    rbc = forms.ChoiceField(choices=[('normal', 'Normal'), ('abnormal', 'Abnormal')], label='Red Blood Cells')
    pc = forms.ChoiceField(choices=[('normal', 'Normal'), ('abnormal', 'Abnormal')], label='Pus Cells')
    pcc = forms.ChoiceField(choices=[('present', 'Present'), ('notpresent', 'Not Present')], label='Pus Cell Clumps')
    ba = forms.ChoiceField(choices=[('present', 'Present'), ('notpresent', 'Not Present')], label='Bacteria')
    htn = forms.ChoiceField(choices=[('yes', 'Yes'), ('no', 'No')], label='Hypertension')
    dm = forms.ChoiceField(choices=[('yes', 'Yes'), ('no', 'No')], label='Diabetes Mellitus')
    cad = forms.ChoiceField(choices=[('yes', 'Yes'), ('no', 'No')], label='Coronary Artery Disease')
    appet = forms.ChoiceField(choices=[('good', 'Good'), ('poor', 'Poor')], label='Appetite')
    pe = forms.ChoiceField(choices=[('yes', 'Yes'), ('no', 'No')], label='Pedal Edema')
    ane = forms.ChoiceField(choices=[('yes', 'Yes'), ('no', 'No')], label='Anemia')
