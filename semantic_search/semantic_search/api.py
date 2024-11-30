import numpy as np
import os
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from search.duplicate_finder import SemanticSearcher


# Initialize FastAPI
app = FastAPI()

# Define request model
class SearchQuery(BaseModel):
    query: str

# Load the CSV file dynamically
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "Corona_NLP_test_small.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    return pd.read_csv(csv_path, encoding='Latin-1')


# Assuming df is a DataFrame, and model_id is a string
df = load_data()
model_id = "paraphrase-multilingual-MiniLM-L12-v2" 
searcher = SemanticSearcher(df=df, model_id=model_id)
searcher.encode_summaries()
searcher.build_vector_database()

@app.post("/api/search")
async def search(query: SearchQuery):
    try:
        results = searcher.search(query.query, k=3)  # Assuming the search method is defined in SemanticSearcher
        # Convert results to a JSON-serializable format
        print(f"Raw search results: {results}")
        # Process results based on their type
        if isinstance(results, list):  # If the results are a list
            results = [
                {key: (None if isinstance(value, float) and np.isnan(value) else value) for key, value in result.items()}
                for result in results
                ]
        elif isinstance(results, pd.DataFrame):  # If the results are a DataFrame
            # Replace NaN values with None in the DataFrame
            results = results.where(pd.notna(results), None)
            results = results.to_dict(orient='records') if not results.empty else None
        else:
            # Handle unexpected types gracefully
            raise ValueError("Unexpected results format from SemanticSearcher.search")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API!"}


@app.get("/favicon.ico")
async def favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    if not os.path.exists(favicon_path):
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(favicon_path)
