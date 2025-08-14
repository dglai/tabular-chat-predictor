import pandas as pd
import numpy as np
import duckdb

def create_table(
    tables: dict[str, pd.DataFrame],
    timestamps: "pd.Series[pd.Timestamp]",
) -> pd.DataFrame:
    comments = tables['comments']
    votes = tables['votes']
    posts = tables['posts']
    users = tables['users']
    timestamp_df = pd.DataFrame({'timestamp': timestamps})
    timedelta = pd.Timedelta(days=365 // 4)
    
    return duckdb.sql(
        f"""
        WITH
        ALL_ENGAGEMENT AS (
            SELECT
                p.id,
                p.owneruserid as userid,
                p.creationdate
            FROM
                posts p
            UNION
            SELECT
                v.id,
                v.userid,
                v.creationdate
            FROM
                votes v
            UNION
            SELECT
                c.id,
                c.userid,
                c.creationdate
            FROM
                comments c
        ),

        ACTIVE_USERS AS (
                SELECT
                t.timestamp,
                u.id,
                count(distinct a.id) as n_engagement
            FROM timestamp_df t
            CROSS JOIN users u
            LEFT JOIN all_engagement a
            ON u.id = a.UserId
                and a.CreationDate <= t.timestamp
            WHERE u.id != -1
            GROUP BY t.timestamp, u.id
        )

        SELECT
            u.timestamp AS __timestamp,
            u.id as __id,
            IF(count(distinct a.id) >= 1, 1, 0) as __label
        FROM
            active_users u
        LEFT JOIN
            all_engagement a
        ON
            u.id = a.UserId AND
            a.CreationDate > u.timestamp AND
            a.CreationDate <= u.timestamp + INTERVAL '{timedelta}'
        where
            u.n_engagement >= 1
        GROUP BY
            u.timestamp, u.id
        ;
        """
    ).df()