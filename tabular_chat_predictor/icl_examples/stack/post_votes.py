import pandas as pd
import numpy as np
import duckdb

def create_table(
    tables: dict[str, pd.DataFrame],
    timestamps: "pd.Series[pd.Timestamp]",
) -> pd.DataFrame:
    votes = tables['votes']
    posts = tables['posts']
    timestamp_df = pd.DataFrame({'timestamp': timestamps})
    timedelta = pd.Timedelta(days=365 // 4)
    
    return duckdb.sql(
        f"""
        SELECT
            t.timestamp AS __timestamp,
            p.id AS __id,
            COUNT(distinct v.id) AS __label
        FROM
            timestamp_df t
        LEFT JOIN
            posts p
        ON
            p.CreationDate <= t.timestamp AND
            p.owneruserid != -1 AND
            p.owneruserid is not null AND
            p.PostTypeId = 1
        LEFT JOIN
            votes v
        ON
            p.id = v.PostId AND
            v.CreationDate > t.timestamp AND
            v.CreationDate <= t.timestamp + INTERVAL '{timedelta}' AND
            v.votetypeid = 2
        GROUP BY
            t.timestamp,
            p.id
        ;
        """
    ).df()