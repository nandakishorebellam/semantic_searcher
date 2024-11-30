# -*- coding: utf-8 -*-

# pylint: disable=line-too-long
"""Module contains implementation of semantic searcher from a csv containing all details of the tweets."""

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearcher:
    """Base class for semantic searcher."""
    def __init__(self, df: pd.DataFrame, model_id: str, target_field: str):
        """Constructor of the class.

        Args:
            df: DataFrame containing tweet data
            model_id: Name of the hugging face model used for multilingual search
            target_field: Field of the dataframe which has to be searched
        """
        self.df = df
        self.encoder = SentenceTransformer(model_id)
        self.target_field = target_field
        # Check if the target_field exists in the dataframe
        if self.target_field not in self.df.columns:
            raise ValueError(f"Target field '{self.target_field}' not found in the uploaded CSV.")

        self.dim = None
        self.index = None
        self.vectors = None

    def encode_summaries(self):
        """Encoding the text into vector embeddings."""
        self.vectors = self.encoder.encode(self.df[self.target_field])
        self.dim = self.vectors.shape[1]

    def build_vector_database(self):
        """Use the Euclidean distance based similarity search."""
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, 3)
        self.index.train(self.vectors)
        self.index.add(self.vectors)

    def search(self, query: str, k: int) -> pd.DataFrame:
        """Search through the database.

        Args:
            query (str): the search query
            k (int): the number of results to return (default 3)

        Returns:
            pd.DataFrame: a dataframe containing the top k search results
        """
        search_query_vec = self.encoder.encode(query)
        svec = np.array(search_query_vec).reshape(1, -1)
        self.index.nprobe = 1
        # pylint: disable=no-value-for-parameter
        _, indices = self.index.search(svec, k)
        return self.df.iloc[indices[0]]
