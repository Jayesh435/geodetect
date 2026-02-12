from django import forms
from .models import ImageUpload


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = ['original_image']
        widgets = {
            'original_image': forms.ClearableFileInput(attrs={
                'accept': 'image/*',
                'class': 'form-control',
            })
        }
        labels = {
            'original_image': 'Select a geotagged image',
        }
