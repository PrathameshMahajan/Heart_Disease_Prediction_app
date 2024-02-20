from django import forms
from .models import HeartDiseaseInput


class HeartDiseaseInputForm(forms.ModelForm):
    class Meta:
        model = HeartDiseaseInput
        fields = '__all__'
