import pandas as pd
import numpy as np
import duckdb

def create_table(
    tables: dict[str, pd.DataFrame],
    timestamps: "pd.Series[pd.Timestamp]",
) -> pd.DataFrame:
    users = tables['users']
    badges = tables['badges']
    timestamp_df = pd.DataFrame({'timestamp': timestamps})
    timedelta = pd.Timedelta(days=365 // 4)
    
    return duckdb.sql(
        f"""
        SELECT
            t.timestamp AS __timestamp,
            u.Id as UserId AS __id,
        CASE WHEN
            COUNT(b.Id) >= 1 THEN 1 ELSE 0 END AS __label
        FROM
            timestamp_df t
        LEFT JOIN
            users u
        ON
            u.CreationDate <= t.timestamp
        LEFT JOIN
            badges b
        ON
            u.Id = b.UserID
            AND b.Date > t.timestamp
            AND b.Date <= t.timestamp + INTERVAL '{timedelta}'
        GROUP BY
            t.timestamp,
            u.Id
        """
    ).df()