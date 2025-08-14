import pandas as pd
import numpy as np

def create_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Sample 5 random rows from the 'review' table and return them as a DataFrame.
    """
    review = tables['review']
    return review.sample(n=5)