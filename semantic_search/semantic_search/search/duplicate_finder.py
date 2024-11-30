# -*- coding: utf-8 -*-

# pylint: disable=line-too-long
"""Module contains implementation of semantic searcher from a csv containing all details of the tweets."""

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearcher:
    """Base class for semantic searcher."""
    def __init__(self, df: pd.DataFrame, model_id: str):
        """Constructor of the class.

        Args:
            df (pd.DataFrame): DataFrame containing tweet data
            model_id (str): name of the hugging face model used for multilingual search
        """
        self.df = df
        self.encoder = SentenceTransformer(model_id)
        self.dim = None
        self.index = None
        self.vectors = None

    def encode_summaries(self):
        """Encoding the text into vector embeddings."""
        self.vectors = self.encoder.encode(self.df['OriginalTweet'])
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