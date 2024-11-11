# -*- coding: utf-8 -*-

# pylint: disable=line-too-long
"""Module contains implementation of duplicate finder from an excel containing all details of the tickets."""

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DuplicateFinder:
    """Base class for duplicate finder."""
    def __init__(self, csv_file: str, model_id: str):
        """Constructor of the class.

        Args:
            csv_file (str): path of the file containing tickets
            model_id (str): name of the hugging face model used for multilingual search
        """
        self.df = pd.read_csv(csv_file, encoding='Latin-1')
        self.encoder = SentenceTransformer(model_id)
        self.dim = None
        self.index = None
        self.vectors = None

    def encode_summaries(self):
        """Encoding the text into vector embeddings."""
        self.vectors = self.encoder.encode(self.df.Summary)
        self.dim = self.vectors.shape[1]

    def build_vector_database(self):
        """Use the Euclidean distance based similarity search."""
        self.index = faiss.IndexFlatL2(self.dim)
        # pylint: disable=no-value-for-parameter
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
        # pylint: disable=no-value-for-parameter
        _, indices = self.index.search(svec, k)
        return self.df.loc[indices[0]]


# pylint: disable=line-too-long
finder = DuplicateFinder(csv_file='JiraSearchRequest.csv', model_id='paraphrase-multilingual-MiniLM-L12-v2')
finder.encode_summaries()
finder.build_vector_database()
results = finder.search(query='Zusammenhang mit der Berichterstattung zusammen', k=3)

# Print search results
print(results)
