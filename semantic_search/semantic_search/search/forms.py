from django import forms

class UploadFileSearchForm(forms.Form):
    csv_file = forms.FileField(label="Upload your CSV File")
    query = forms.CharField(label="Enter your search query", max_length=500)
