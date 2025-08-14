#!/usr/bin/env python3
"""
RDB to Tab2Graph DFS Dataset Conversion Script
"""

import os
import yaml
import pandas as pd
import numpy as np
import argparse
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tqdm
import duckdb

class DatasetConverter:
    def __init__(self, sampling_rate=0.3, activity_bias_factor=2.0):
        self.sampling_rate = sampling_rate
        self.activity_bias_factor = activity_bias_factor
        
    def convert(self, input_folder: str, output_folder: str, today_date: str = None):
        """Main conversion function."""
        # Step 1: Load input data
        metadata = self._load_metadata(input_folder)
        entity_tables = self._load_entity_tables(input_folder)
        timestamps = self._generate_timestamps(input_folder, entity_tables, metadata, today_date)
        
        # Step 2: Analyze schema
        target_tables = self._identify_target_tables(metadata)
        relationships = self._map_foreign_keys(metadata)
        strategies = self._determine_strategies(metadata, relationships, target_tables)
        
        # Step 3: Process each target table
        tasks = []
        for table_name in tqdm.tqdm(target_tables, position=0, desc="Processing tables"):
            entity_table = entity_tables[f"{table_name}.pqt"]
            strategy = strategies[table_name]
            
            if strategy == "static":
                training_data = self._process_static_table(entity_table, metadata, table_name)
            else:
                training_data = self._process_temporal_table(
                    entity_table, timestamps, strategy, metadata, table_name,
                    entity_tables, relationships
                )
            
            task = self._create_task_metadata(table_name, training_data, metadata)
            tasks.append(task)
            self._write_task_output(output_folder, table_name, training_data)
        
        # Step 4: Generate output
        self._copy_entity_tables(input_folder, output_folder)
        self._update_metadata_with_tasks(output_folder, metadata, tasks)
        
        return tasks
    
    def _load_metadata(self, input_folder: str):
        with open(os.path.join(input_folder, "metadata.yaml"), 'r') as f:
            return yaml.safe_load(f)
    
    def _load_entity_tables(self, input_folder: str):
        tables = {}
        for file in os.listdir(input_folder):
            if file.endswith('.pqt'):
                tables[file] = pd.read_parquet(os.path.join(input_folder, file))
        return tables
    
    def _generate_timestamps(self, input_folder: str, entity_tables=None, metadata=None, today_date=None):
        # Generate timestamps automatically using today_date
        if today_date and entity_tables and metadata:
            earliest_date = self._find_earliest_date_in_dataset(entity_tables, metadata)
            if earliest_date:
                generated_timestamps = self._generate_exponential_timestamps(today_date, earliest_date)
                return generated_timestamps
        
        return []
    
    def _find_earliest_date_in_dataset(self, entity_tables, metadata):
        """Find the earliest date across all temporal columns in the dataset."""
        earliest_date = None
        
        for table in metadata['tables']:
            time_column = table.get('time_column')
            if time_column:
                table_file = f"{table['name']}.pqt"
                if table_file in entity_tables:
                    df = entity_tables[table_file]
                    if time_column in df.columns:
                        # Convert to datetime if not already
                        dates = pd.to_datetime(df[time_column])
                        min_date = dates.min()
                        if earliest_date is None or min_date < earliest_date:
                            earliest_date = min_date
        
        return earliest_date
    
    def _generate_exponential_timestamps(self, today_date, earliest_date):
        """Generate timestamps with exponential backoff intervals."""
        timestamps = []
        current_date = datetime.strptime(today_date, '%Y-%m-%d')
        day_interval = 1
        count_in_current_interval = 0
        
        while current_date >= earliest_date:
            timestamps.append(current_date)
            current_date -= timedelta(days=day_interval)
            count_in_current_interval += 1
            
            # Every 7 timestamps, double the interval
            if count_in_current_interval == 7:
                day_interval *= 2
                count_in_current_interval = 0
        
        return timestamps
    
    def _identify_target_tables(self, metadata):
        target_tables = []
        for table in metadata['tables']:
            has_primary_key = any(col['dtype'] == 'primary_key' for col in table['columns'])
            if has_primary_key:
                target_tables.append(table['name'])
        return target_tables
    
    def _map_foreign_keys(self, metadata):
        relationships = {}
        for table in metadata['tables']:
            table_name = table['name']
            relationships[table_name] = []
            for col in table['columns']:
                if col['dtype'] == 'foreign_key':
                    target_table, target_column = col['link_to'].split('.')
                    relationships[table_name].append({
                        'foreign_key_column': col['name'],
                        'target_table': target_table,
                        'target_column': target_column
                    })
        return relationships
    
    def _determine_strategies(self, metadata, relationships, target_tables):
        strategies = {}
        table_lookup = {table['name']: table for table in metadata['tables']}
        
        for table_name in target_tables:
            table = table_lookup[table_name]
            has_temporal = table.get('time_column') is not None
            
            if has_temporal:
                strategies[table_name] = "creation_based"
            else:
                # Check if any tables reference this table and have temporal columns
                has_temporal_references = False
                for other_table_name, other_table in table_lookup.items():
                    if other_table_name != table_name and other_table.get('time_column'):
                        for rel in relationships.get(other_table_name, []):
                            if rel['target_table'] == table_name:
                                has_temporal_references = True
                                break
                        if has_temporal_references:
                            break
                
                strategies[table_name] = "activity_based" if has_temporal_references else "static"
        
        return strategies
    
    def _process_static_table(self, entity_table, metadata, table_name):
        """Process static table using vectorized operations."""
        table_meta = next(table for table in metadata['tables'] if table['name'] == table_name)
        
        # Get columns to include (exclude foreign keys and temporal columns)
        include_cols = []
        for col in table_meta['columns']:
            if (col['dtype'] not in ['foreign_key'] and 
                col['name'] != table_meta.get('time_column')):
                include_cols.append(col['name'])
        
        # Create training data using vectorized operations
        training_df = entity_table[include_cols].copy()
        training_df['__label__'] = np.random.choice([0, 1], size=len(training_df))
        
        return training_df
    
    def _process_temporal_table(self, entity_table, timestamps, strategy, metadata, table_name,
                              entity_tables, relationships):
        """Process temporal table using DuckDB optimization."""
        table_meta = next(table for table in metadata['tables'] if table['name'] == table_name)
        
        # Use DuckDB-optimized processing
        return self._process_temporal_table_duckdb(
            entity_table, timestamps, strategy, metadata, table_name, entity_tables, relationships
        )
    
    def _process_temporal_table_duckdb(self, entity_table, timestamps, strategy, metadata, table_name,
                                     entity_tables, relationships):
        """DuckDB-optimized temporal table processing for all timestamps at once."""
        table_meta = next(table for table in metadata['tables'] if table['name'] == table_name)
        
        # Create DuckDB connection
        conn = duckdb.connect()
        
        if strategy == "creation_based":
            result = self._process_creation_based_duckdb(conn, entity_table, timestamps, table_meta)
        else:  # activity_based
            result = self._process_activity_based_duckdb(
                conn, entity_table, timestamps, table_meta, entity_tables, relationships, metadata
            )
        conn.close()
        return result
    
    
    def _get_primary_key_column(self, table_meta):
        for col in table_meta['columns']:
            if col['dtype'] == 'primary_key':
                return col['name']
    
    def _get_include_columns(self, table_meta):
        """Get columns to include in training data (exclude foreign keys and time columns)."""
        include_cols = []
        for col in table_meta['columns']:
            if (col['dtype'] not in ['foreign_key'] and
                col['name'] != table_meta.get('time_column')):
                include_cols.append(col['name'])
        return include_cols
    
    def _process_creation_based_duckdb(self, conn, entity_table, timestamps, table_meta):
        """Process creation-based strategy using DuckDB for all timestamps at once."""
        temporal_column = table_meta['time_column']
        include_cols = self._get_include_columns(table_meta)
        
        # Register tables with DuckDB
        conn.register('entity_table', entity_table)
        timestamps_df = pd.DataFrame({'__timestamp__': timestamps})
        conn.register('timestamps', timestamps_df)
        
        # Build column selection for SQL
        include_cols_sql = ', '.join([f'e.{col}' for col in include_cols])
        
        # Single SQL query to process all timestamps at once
        sql = f"""
        WITH eligible_entities AS (
            SELECT
                t.__timestamp__,
                {include_cols_sql},
                ROW_NUMBER() OVER (PARTITION BY t.__timestamp__ ORDER BY RANDOM()) as rn,
                COUNT(*) OVER (PARTITION BY t.__timestamp__) as total_count
            FROM timestamps t
            CROSS JOIN entity_table e
            WHERE e.{temporal_column} <= t.__timestamp__
        ),
        sampled_entities AS (
            SELECT *
            FROM eligible_entities
            WHERE rn <= GREATEST(1, CAST(total_count * {self.sampling_rate} AS INTEGER))
        )
        SELECT
            __timestamp__,
            {', '.join(include_cols)},
            CAST(RANDOM() < 0.5 AS INTEGER) as __label__
        FROM sampled_entities
        ORDER BY __timestamp__, {include_cols[0] if include_cols else '__timestamp__'}
        """
        
        return conn.execute(sql).df()
    
    def _process_activity_based_duckdb(self, conn, entity_table, timestamps, table_meta,
                                     entity_tables, relationships, metadata):
        """Process activity-based strategy using DuckDB for all timestamps at once."""
        primary_key = self._get_primary_key_column(table_meta)
        include_cols = self._get_include_columns(table_meta)
        table_lookup = {table['name']: table for table in metadata['tables']}
        
        # Register all tables with DuckDB
        conn.register('entity_table', entity_table)
        timestamps_df = pd.DataFrame({'__timestamp__': timestamps})
        conn.register('timestamps', timestamps_df)
        
        for table_file, df in entity_tables.items():
            table_alias = table_file.replace('.pqt', '')
            conn.register(table_alias, df)
        
        # Build activity tracking subqueries
        activity_unions = []
        
        for other_table_name, other_table_meta in table_lookup.items():
            if other_table_meta.get('time_column') and other_table_name != table_meta['name']:
                for rel in relationships.get(other_table_name, []):
                    if rel['target_table'] == table_meta['name']:
                        temporal_col = other_table_meta['time_column']
                        foreign_key_col = rel['foreign_key_column']
                        
                        activity_unions.append(f"""
                        SELECT
                            t.__timestamp__,
                            r.{foreign_key_col} as entity_id,
                            COUNT(*) as activity_count
                        FROM timestamps t
                        JOIN {other_table_name} r ON r.{temporal_col} <= t.__timestamp__
                        GROUP BY t.__timestamp__, r.{foreign_key_col}
                        """)
        
        if not activity_unions:
            # No activity data available, return empty DataFrame
            return pd.DataFrame()
        
        # Combine all activity data
        activity_sql = " UNION ALL ".join(activity_unions)
        
        # Build column selection for SQL
        include_cols_sql = ', '.join([f'e.{col}' for col in include_cols])
        
        # Main query with activity-based sampling
        sql = f"""
        WITH activity_data AS (
            {activity_sql}
        ),
        entity_activity AS (
            SELECT
                __timestamp__,
                entity_id,
                SUM(activity_count) as total_activity
            FROM activity_data
            GROUP BY __timestamp__, entity_id
        ),
        eligible_entities AS (
            SELECT
                ea.__timestamp__,
                {include_cols_sql},
                COALESCE(ea.total_activity, 1) as activity_score,
                ROW_NUMBER() OVER (
                    PARTITION BY ea.__timestamp__
                    ORDER BY POWER(COALESCE(ea.total_activity, 1), {self.activity_bias_factor}) * RANDOM() DESC
                ) as rn,
                COUNT(*) OVER (PARTITION BY ea.__timestamp__) as total_count
            FROM entity_activity ea
            JOIN entity_table e ON e.{primary_key} = ea.entity_id
        ),
        sampled_entities AS (
            SELECT *
            FROM eligible_entities
            WHERE rn <= GREATEST(1, CAST(total_count * {self.sampling_rate} AS INTEGER))
        )
        SELECT
            __timestamp__,
            {', '.join(include_cols)},
            CAST(RANDOM() < 0.5 AS INTEGER) as __label__
        FROM sampled_entities
        ORDER BY __timestamp__, {include_cols[0] if include_cols else '__timestamp__'}
        """
        
        return conn.execute(sql).df()
    
    def _create_task_metadata(self, table_name, training_data, metadata):
        """Create task metadata from training DataFrame."""
        if isinstance(training_data, pd.DataFrame):
            if len(training_data) == 0:
                raise ValueError(f"No training data generated for table {table_name}")
            sample_cols = training_data.columns.tolist()
        else:
            sample_cols = list(training_data[0].keys()) if training_data else []
        
        table_meta = next(table for table in metadata['tables'] if table['name'] == table_name)
        primary_key = self._get_primary_key_column(table_meta)
        
        columns = []
        if '__timestamp__' in sample_cols:
            columns.append({'dtype': 'datetime', 'name': '__timestamp__'})
        
        columns.append({'dtype': 'primary_key', 'name': primary_key})
        columns.append({'dtype': 'category', 'name': '__label__'})
        
        for col in table_meta['columns']:
            col_name = col['name']
            if (col_name in sample_cols and
                col_name != primary_key and
                col_name not in ['__timestamp__', '__label__'] and
                col['dtype'] not in ['foreign_key'] and
                col_name != table_meta.get('time_column')):
                columns.append({'dtype': col['dtype'], 'name': col_name})
        
        return {
            'name': f"{table_name}-template",
            'target_table': table_name,
            'task_type': 'classification',
            'target_column': '__label__',
            'time_column': '__timestamp__' if '__timestamp__' in sample_cols else None,
            'evaluation_metric': 'auroc',
            'format': 'parquet',
            'source': f"{table_name}-template/{{split}}.pqt",
            'columns': columns,
            'key_prediction_label_column': 'label',
            'key_prediction_query_idx_column': 'query_idx',
            'task_emb': None
        }
    
    def _write_task_output(self, output_folder, table_name, training_data):
        """Write task output files."""
        task_dir = os.path.join(output_folder, f"{table_name}-template")
        os.makedirs(task_dir, exist_ok=True)
        
        if isinstance(training_data, pd.DataFrame) and len(training_data) > 0:
            training_data.to_parquet(os.path.join(task_dir, "train.pqt"), index=False)
        elif isinstance(training_data, list) and training_data:
            training_data = pd.DataFrame(training_data)
            training_data.to_parquet(os.path.join(task_dir, "train.pqt"), index=False)
        
        # Create empty validation and test files
        empty_df = training_data.iloc[0:0].copy()  # Keep the same schema
        empty_df.to_parquet(os.path.join(task_dir, "validation.pqt"), index=False)
        empty_df.to_parquet(os.path.join(task_dir, "test.pqt"), index=False)
    
    def _copy_entity_tables(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for file in os.listdir(input_folder):
            if file.endswith('.pqt'):
                shutil.copy2(os.path.join(input_folder, file), 
                           os.path.join(output_folder, file))
    
    def _update_metadata_with_tasks(self, output_folder, metadata, tasks):
        metadata['tasks'] = tasks
        with open(os.path.join(output_folder, "metadata.yaml"), 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description='Convert RDB dataset to Tab2Graph DFS format')
    parser.add_argument('input_folder', help='Path to input RDB folder')
    parser.add_argument('output_folder', help='Path to output DFS folder')
    parser.add_argument('--sampling_rate', type=float, default=0.3, help='Sampling rate')
    parser.add_argument('--activity_bias_factor', type=float, default=2.0, help='Activity bias factor')
    parser.add_argument('--today_date', type=str, help='Today\'s date for automatic timestamp generation (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    
    converter = DatasetConverter(args.sampling_rate, args.activity_bias_factor)
    tasks = converter.convert(args.input_folder, args.output_folder, args.today_date)
    
    print(f"Conversion completed. Generated {len(tasks)} tasks:")
    for task in tasks:
        print(f"  - {task['name']} ({task['target_table']})")


if __name__ == "__main__":
    main()