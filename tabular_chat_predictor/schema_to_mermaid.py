"""
Helper function to convert schema YAML files to Mermaid ER diagrams.

This module provides functionality to parse database schema YAML files
and generate Mermaid ER diagram syntax for visualization.
"""

import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path


def schema_yaml_to_mermaid(schema_file: str, title: Optional[str] = None) -> str:
    """
    Convert a schema YAML file to Mermaid ER diagram syntax.
    
    Args:
        schema_file (str): Path to the YAML schema file
        title (Optional[str]): Optional title for the diagram
        
    Returns:
        str: Mermaid ER diagram syntax
        
    Raises:
        FileNotFoundError: If the schema file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        KeyError: If required schema fields are missing
    """
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")
    
    return schema_dict_to_mermaid(schema, title)


def schema_dict_to_mermaid(schema: Dict[str, Any], title: Optional[str] = None) -> str:
    """
    Convert a schema dictionary to Mermaid ER diagram syntax.
    
    Args:
        schema (Dict[str, Any]): Schema dictionary with 'tables' key
        title (Optional[str]): Optional title for the diagram
        
    Returns:
        str: Mermaid ER diagram syntax
        
    Raises:
        KeyError: If required schema fields are missing
    """
    if 'tables' not in schema:
        raise KeyError("Schema must contain 'tables' key")
    
    mermaid_lines = []
    
    mermaid_lines.append("erDiagram")
    
    # Process tables and collect relationships
    relationships = []
    
    for table in schema['tables']:
        if 'name' not in table or 'columns' not in table:
            continue
            
        table_name = table['name']
        columns = table['columns']
        
        # Add table definition
        mermaid_lines.append(f"    {table_name} {{")
        
        for column in columns:
            if 'name' not in column or 'dtype' not in column:
                continue
                
            col_name = column['name']
            col_type = column['dtype']
            
            # Format column based on type
            if col_type == 'primary_key':
                mermaid_lines.append(f"        int {col_name} PK")
            elif col_type == 'foreign_key':
                mermaid_lines.append(f"        int {col_name} FK")
                
                # Collect relationship information
                if 'link_to' in column:
                    link_target = column['link_to']
                    if '.' in link_target:
                        target_table = link_target.split('.')[0]
                        relationships.append((target_table, table_name, col_name))
            else:
                # Map other data types to Mermaid types
                mermaid_type = _map_dtype_to_mermaid(col_type)
                mermaid_lines.append(f"        {mermaid_type} {col_name}")
        
        mermaid_lines.append("    }")
    
    # Add relationships
    if relationships:
        mermaid_lines.append("")
        for parent_table, child_table, fk_column in relationships:
            # Use one-to-many relationship notation
            mermaid_lines.append(f"    {parent_table} ||--o{{ {child_table} : \"{fk_column}\"")
    
    # Close mermaid block if title was added
    if title:
        mermaid_lines.append("```")
    
    return "\n".join(mermaid_lines)


def _map_dtype_to_mermaid(dtype: str) -> str:
    """
    Map schema data types to appropriate Mermaid ER diagram types.
    
    Args:
        dtype (str): Data type from schema
        
    Returns:
        str: Corresponding Mermaid type
    """
    type_mapping = {
        'datetime': 'datetime',
        'float': 'float',
        'int': 'int',
        'integer': 'int', 
        'string': 'string',
        'text': 'string',
        'category': 'string',
        'boolean': 'boolean',
        'bool': 'boolean'
    }
    
    return type_mapping.get(dtype.lower(), 'string')


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Convert Amazon schema
        amazon_mermaid = schema_yaml_to_mermaid(
            "icl_examples/amazon/schema.yaml", 
            "Amazon E-commerce Database Schema"
        )
        print("Amazon Schema Mermaid Diagram:")
        print(amazon_mermaid)
        print("\n" + "="*50 + "\n")
        
        # Convert Stack Exchange schema  
        stack_mermaid = schema_yaml_to_mermaid(
            "icl_examples/stack/schema.yaml",
            "Stack Exchange Database Schema"
        )
        print("Stack Exchange Schema Mermaid Diagram:")
        print(stack_mermaid)
        
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        print(f"Error: {e}")