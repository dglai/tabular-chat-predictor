import pandas as pd
import numpy as np

def create_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Find the post with the latest timestamp in the posts table."""
    posts = tables['posts']
    latest_post = posts.loc[posts['CreationDate'].idxmax()]
    return pd.DataFrame([latest_post])