from typing import List, Optional, Annotated, get_origin, get_args, Union, Any, Literal
from enum import Enum
import inspect
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


def function_to_json_schema(func) -> dict:
    """
    Convert a Python function with annotations and docstring into an enhanced JSON schema.
    
    This generates schemas compatible with the new format that includes:
    - $defs for complex type definitions
    - anyOf for Union types and optional fields
    - Enhanced metadata with titles and descriptions
    - Better handling of complex nested types
    
    Args:
        func: A Python function with type annotations
        
    Returns:
        dict: Enhanced JSON schema representation of the function
    """
    
    def _get_field_info(annotation):
        """Extract Field information from Annotated types."""
        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            base_type = args[0]
            for metadata in args[1:]:
                if isinstance(metadata, FieldInfo):
                    return base_type, metadata
            return base_type, None
        return annotation, None
    
    def _is_optional(annotation):
        """Check if a type annotation is Optional (Union[T, None])."""
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            return len(args) == 2 and type(None) in args
        return False
    
    def _is_union(annotation):
        """Check if a type annotation is a Union (but not Optional)."""
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            # Not optional if it doesn't contain None or has more than 2 types
            return not (len(args) == 2 and type(None) in args)
        return False
    
    def _get_non_none_type(annotation):
        """Extract the non-None type from Optional[T]."""
        if _is_optional(annotation):
            args = get_args(annotation)
            return next(arg for arg in args if arg is not type(None))
        return annotation
    
    def _get_union_types(annotation):
        """Extract all types from a Union."""
        if get_origin(annotation) is Union:
            return get_args(annotation)
        return (annotation,)
    
    def _type_to_json_schema(type_annotation, field_info=None, defs=None):
        """Convert a Python type to JSON schema with enhanced features."""
        if defs is None:
            defs = {}
            
        # Handle Annotated types first
        if get_origin(type_annotation) is Annotated:
            base_type, extracted_field = _get_field_info(type_annotation)
            if extracted_field:
                field_info = extracted_field
            return _type_to_json_schema(base_type, field_info, defs)
        
        # Handle Optional types (Union[T, None])
        if _is_optional(type_annotation):
            inner_type = _get_non_none_type(type_annotation)
            inner_schema = _type_to_json_schema(inner_type, field_info, defs)
            
            # Create anyOf with null
            schema = {
                "anyOf": [
                    inner_schema,
                    {"type": "null"}
                ],
                "default": None
            }
            
            # Add title from field_info if available
            if field_info and hasattr(field_info, 'title') and field_info.title:
                schema["title"] = field_info.title
                
            return schema
        
        # Handle Union types (non-optional)
        if _is_union(type_annotation):
            union_types = _get_union_types(type_annotation)
            any_of_schemas = []
            
            for union_type in union_types:
                # For Union types, we need to extract field info for each type
                if get_origin(union_type) is Annotated:
                    base_type, type_field_info = _get_field_info(union_type)
                    type_schema = _type_to_json_schema(base_type, type_field_info, defs)
                else:
                    type_schema = _type_to_json_schema(union_type, None, defs)
                any_of_schemas.append(type_schema)
            
            schema = {"anyOf": any_of_schemas}
            
            # Add title from main field_info if available
            if field_info and hasattr(field_info, 'title') and field_info.title:
                schema["title"] = field_info.title
                
            return schema
        
        # Basic types
        if type_annotation is int:
            schema = {"type": "integer"}
        elif type_annotation is str:
            schema = {"type": "string"}
        elif type_annotation is float:
            schema = {"type": "number"}
        elif type_annotation is bool:
            schema = {"type": "boolean"}
        # List types
        elif get_origin(type_annotation) is list or type_annotation is list:
            if get_origin(type_annotation) is list:
                item_type = get_args(type_annotation)[0]
                schema = {
                    "type": "array",
                    "items": _type_to_json_schema(item_type, None, defs)
                }
            else:
                schema = {"type": "array"}
        # Enum types
        elif inspect.isclass(type_annotation) and issubclass(type_annotation, Enum):
            if issubclass(type_annotation, str):
                # Add enum to defs
                enum_name = type_annotation.__name__
                enum_def = {
                    "type": "string",
                    "enum": [e.value for e in type_annotation],
                    "title": enum_name
                }
                
                # Add description from docstring if available
                if type_annotation.__doc__:
                    enum_def["description"] = type_annotation.__doc__.strip()
                
                defs[enum_name] = enum_def
                
                # Return reference
                schema = {"$ref": f"#/$defs/{enum_name}"}
            else:
                raise ValueError(f"Only string-based enums are supported, got {type_annotation}")
        # Literal types
        elif get_origin(type_annotation) is Literal:
            literal_values = get_args(type_annotation)
            if not literal_values:
                raise ValueError(f"Literal type must have at least one value")
            
            # Check if all values are strings
            if all(isinstance(val, str) for val in literal_values):
                schema = {
                    "type": "string",
                    "enum": list(literal_values)
                }
            else:
                raise ValueError(f"Only string literals are supported, got: {literal_values}")
        # Pydantic BaseModel
        elif inspect.isclass(type_annotation) and issubclass(type_annotation, BaseModel):
            model_name = type_annotation.__name__
            model_def = _basemodel_to_json_schema(type_annotation, defs)
            defs[model_name] = model_def
            
            # Return reference
            schema = {"$ref": f"#/$defs/{model_name}"}
        else:
            raise ValueError(f"Unsupported type: {type_annotation}")
        
        # Add metadata from field_info
        if field_info:
            if hasattr(field_info, 'description') and field_info.description:
                schema["description"] = field_info.description
            if hasattr(field_info, 'title') and field_info.title:
                schema["title"] = field_info.title
            
        return schema
    
    def _basemodel_to_json_schema(model_class, defs=None):
        """Convert a Pydantic BaseModel to JSON schema definition."""
        if defs is None:
            defs = {}
            
        properties = {}
        required = []
        
        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation
            
            # Check if field is required
            is_required = True
            if _is_optional(field_type):
                is_required = False
            elif hasattr(field_info, 'default') and field_info.default not in (..., None, PydanticUndefined):
                is_required = False
            elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                is_required = False
            
            if is_required:
                required.append(field_name)
            
            # Convert field type to schema
            field_schema = _type_to_json_schema(field_type, field_info, defs)
            properties[field_name] = field_schema
        
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "title": model_class.__name__
        }
        
        # Add description from docstring if available
        if model_class.__doc__:
            schema["description"] = model_class.__doc__.strip()
            
        return schema
    
    # Get function signature and docstring
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Build properties, required list, and defs
    properties = {}
    required = []
    defs = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':  # Skip self parameter
            continue
            
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            raise ValueError(f"Parameter {param_name} missing type annotation")
        
        # Extract field info from Annotated types
        base_type, field_info = _get_field_info(annotation)
        
        # Check if parameter is required
        is_required = True
        
        # Check if it's Optional
        if _is_optional(annotation) or _is_optional(base_type):
            is_required = False
        
        # Check if it has a default value
        if param.default != inspect.Parameter.empty:
            is_required = False
        
        # Check if Field has default or default_factory
        if field_info:
            if hasattr(field_info, 'default') and field_info.default not in (..., None, PydanticUndefined):
                is_required = False
            elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                is_required = False
        
        if is_required:
            required.append(param_name)
        
        # Convert type to JSON schema
        param_schema = _type_to_json_schema(annotation, field_info, defs)
        
        # Add title if not already present
        if "title" not in param_schema:
            # Create title from parameter name
            param_schema["title"] = param_name.replace('_', ' ').title()
        
        properties[param_name] = param_schema
    
    # Build the complete schema
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
    
    # Add $defs if we have any
    if defs:
        schema["function"]["parameters"]["$defs"] = defs
    
    return schema


class Tool:
    """
    A callable tool that wraps a Python function with enhanced JSON schema conversion.
    
    This version supports the new enhanced schema format with:
    - $defs for complex type definitions
    - anyOf for Union types and optional fields
    - Enhanced metadata handling
    - Better parsing of complex nested types
    
    Attributes:
        schema: Enhanced JSON schema representation of the function
        func: The original Python function
    """
    
    def __init__(self, func):
        self.func = func
        self.schema = function_to_json_schema(func)
        self._sig = inspect.signature(func)
        self._param_types = self._cache_parameter_types()
        self._defs = self.schema.get("function", {}).get("parameters", {}).get("$defs", {})
    
    def _cache_parameter_types(self) -> dict:
        """Cache parameter type information for efficient parsing."""
        param_types = {}
        
        for param_name, param in self._sig.parameters.items():
            if param_name == 'self':
                continue
            
            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                raise ValueError(f"Parameter {param_name} missing type annotation")
            
            param_types[param_name] = {
                'annotation': annotation,
                'has_default': param.default != inspect.Parameter.empty
            }
        
        return param_types
    
    def _get_field_info(self, annotation):
        """Extract Field information from Annotated types."""
        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            base_type = args[0]
            return base_type
        return annotation
    
    def _is_optional(self, annotation):
        """Check if a type annotation is Optional (Union[T, None])."""
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            return len(args) == 2 and type(None) in args
        return False
    
    def _is_union(self, annotation):
        """Check if a type annotation is a Union (but not Optional)."""
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            # Not optional if it doesn't contain None or has more than 2 types
            return not (len(args) == 2 and type(None) in args)
        return False
    
    def _get_non_none_type(self, annotation):
        """Extract the non-None type from Optional[T]."""
        if self._is_optional(annotation):
            args = get_args(annotation)
            return next(arg for arg in args if arg is not type(None))
        return annotation
    
    def _get_union_types(self, annotation):
        """Extract all types from a Union."""
        if get_origin(annotation) is Union:
            return get_args(annotation)
        return (annotation,)
    
    def _resolve_ref(self, ref_string):
        """Resolve a $ref to its definition."""
        if ref_string.startswith("#/$defs/"):
            def_name = ref_string[8:]  # Remove "#/$defs/"
            if def_name in self._defs:
                return self._defs[def_name]
        raise ValueError(f"Cannot resolve reference: {ref_string}")
    
    def _parse_value(self, value: Any, type_annotation: Any) -> Any:
        """Parse a JSON value according to the given type annotation with enhanced Union support."""
        # Handle None values
        if value is None:
            if self._is_optional(type_annotation):
                return None
            raise ValueError(f"None value provided for non-optional type {type_annotation}")
        
        # Handle Optional types
        if self._is_optional(type_annotation):
            inner_type = self._get_non_none_type(type_annotation)
            return self._parse_value(value, inner_type)
        
        # Handle Union types (non-optional)
        if self._is_union(type_annotation):
            union_types = self._get_union_types(type_annotation)
            
            # Try each type in the union until one works
            for union_type in union_types:
                try:
                    return self._parse_value(value, union_type)
                except (ValueError, TypeError):
                    continue
            
            # If none worked, raise an error
            type_names = [getattr(t, '__name__', str(t)) for t in union_types]
            raise ValueError(f"Value {value} doesn't match any type in Union[{', '.join(type_names)}]")
        
        # Handle Annotated types
        if get_origin(type_annotation) is Annotated:
            base_type = self._get_field_info(type_annotation)
            return self._parse_value(value, base_type)
        
        # Basic types - direct mapping
        if type_annotation is int:
            return int(value)
        elif type_annotation is str:
            return str(value)
        elif type_annotation is float:
            return float(value)
        elif type_annotation is bool:
            return bool(value)
        
        # List types
        elif get_origin(type_annotation) is list or type_annotation is list:
            if not isinstance(value, list):
                raise ValueError(f"Expected list, got {type(value)}")
            
            if get_origin(type_annotation) is list:
                item_type = get_args(type_annotation)[0]
                return [self._parse_value(item, item_type) for item in value]
            else:
                return list(value)
        
        # Enum types
        elif inspect.isclass(type_annotation) and issubclass(type_annotation, Enum):
            if isinstance(value, str):
                # Find enum member by value
                for enum_member in type_annotation:
                    if enum_member.value == value:
                        return enum_member
                raise ValueError(f"Invalid enum value '{value}' for {type_annotation}")
            else:
                raise ValueError(f"Expected string for enum {type_annotation}, got {type(value)}")
        
        # Literal types
        elif get_origin(type_annotation) is Literal:
            literal_values = get_args(type_annotation)
            if isinstance(value, str) and value in literal_values:
                return value
            else:
                raise ValueError(f"Invalid literal value '{value}'. Must be one of: {literal_values}")
        
        # Pydantic BaseModel
        elif inspect.isclass(type_annotation) and issubclass(type_annotation, BaseModel):
            if isinstance(value, dict):
                return type_annotation(**value)
            else:
                raise ValueError(f"Expected dict for BaseModel {type_annotation}, got {type(value)}")
        
        else:
            raise ValueError(f"Unsupported type: {type_annotation}")
    
    def parse(self, json_args_dict: dict) -> dict:
        """
        Parse a JSON argument dict into a Python kwargs dict with enhanced Union support.
        
        Args:
            json_args_dict: Dictionary with JSON-compatible values
            
        Returns:
            dict: Python kwargs dictionary with properly typed values
        """
        kwargs = {}
        
        for param_name, param_info in self._param_types.items():
            if param_name in json_args_dict:
                # Parse the value according to its type
                raw_value = json_args_dict[param_name]
                parsed_value = self._parse_value(raw_value, param_info['annotation'])
                kwargs[param_name] = parsed_value
            elif not param_info['has_default'] and not self._is_optional(param_info['annotation']):
                # Required parameter is missing
                raise ValueError(f"Required parameter '{param_name}' is missing")
            # If parameter has default or is optional and missing, omit it entirely
        
        return kwargs
    
    def __call__(self, json_args_dict: dict) -> Any:
        """
        Call the wrapped function with parsed JSON arguments.
        
        Args:
            json_args_dict: Dictionary with JSON-compatible values
            
        Returns:
            The result of calling the wrapped function
        """
        kwargs = self.parse(json_args_dict)
        return self.func(**kwargs)


def make_tool(func) -> Tool:
    """
    Convert a Python function into a callable Tool object with enhanced schema support.
    
    Args:
        func: A Python function with type annotations
        
    Returns:
        Tool: A Tool object that can parse JSON arguments and call the function
    """
    return Tool(func)