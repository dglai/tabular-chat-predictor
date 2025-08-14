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
            customer_id AS __id,
            ltv AS __label
        FROM
            timestamp_df,
            customer,
            (
                SELECT
                    COALESCE(SUM(price), 0) as ltv,
                FROM
                    review,
                    product
                WHERE
                    review.customer_id = customer.customer_id AND
                    review.product_id = product.product_id AND
                    review_time > timestamp AND
                    review_time <= timestamp + INTERVAL '{timedelta}'
            )
        WHERE
            EXISTS (
                SELECT 1
                FROM review
                WHERE
                    review.customer_id = customer.customer_id AND
                    review_time > timestamp - INTERVAL '{timedelta}' AND
                    review_time <= timestamp
            )
        """
    ).df()