"""Django framework for the semantic searcher."""
from django.shortcuts import render
import pandas as pd

from .forms import UploadFileSearchForm
from .duplicate_finder import SemanticSearcher


def upload_and_search(request):
    """Upload file and search."""
    results = None
    if request.method == 'POST':
        form = UploadFileSearchForm(request.POST, request.FILES)
        if form.is_valid():
            # Read the uploaded file into a DataFrame
            csv_file = request.FILES['csv_file']
            query = form.cleaned_data['query']
            target_field: str = form.cleaned_data['target_field']

            # Convert the in-memory file to a DataFrame
            df: pd.DataFrame = pd.read_csv(csv_file, encoding='Latin-1')
            searcher = SemanticSearcher(df, 'paraphrase-multilingual-MiniLM-L12-v2', target_field)
            searcher.encode_summaries()
            searcher.build_vector_database()
            results = searcher.search(query, k=3)
            results = results.to_dict(orient='records') if not results.empty else None
    else:
        form = UploadFileSearchForm()

    return render(request, 'search/search.html', {'form': form, 'results': results})
