import pandas as pd
import numpy as np
import duckdb

def create_table(
    tables: dict[str, pd.DataFrame],
    timestamps: "pd.Series[pd.Timestamp]",
) -> pd.DataFrame:
    review = tables['review']
    customer = tables['customer']
    product = tables['product']
    timestamp_df = pd.DataFrame({'timestamp': timestamps})
    timedelta = pd.Timedelta(days=365 // 4)
    
    return duckdb.sql(
        f"""
        SELECT
            timestamp AS __timestamp,
            product.product_id AS __id,
            COALESCE(SUM(price), 0) AS __label
        FROM
            timestamp_df,
            product,
            review
        WHERE
            review.product_id = product.product_id AND
            review_time > timestamp AND
            review_time <= timestamp + INTERVAL '{timedelta}'
        GROUP BY
            timestamp,
            product.product_id
        """
    ).df()