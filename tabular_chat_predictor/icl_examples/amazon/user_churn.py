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
            CAST(
                NOT EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review_time > timestamp AND
                        review_time <= timestamp + INTERVAL '{timedelta}'
                ) AS INTEGER
            ) AS __label
        FROM
            timestamp_df,
            customer,
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