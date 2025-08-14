"""
PredictingAssistant for handling predictive query workflows using ComputeEngine.

This module contains the PredictingAssistant class that handles the complete
predictive workflow using ComputeEngine for computational operations and BaseAgent
for common functionality. Includes ML tools previously in FunctionExecutor.
"""

import json
import uuid
import time
from typing import Dict, Any, List, Annotated, Literal
from pathlib import Path
import pandas as pd
import numpy as np
from jinja2 import Template
from pydantic import Field

from .base_agent import BaseAgent
from ..compute.compute_engine import ComputeEngine
from ..core.protocols import ChatUserInterface, ChatLogger
from ..core.tool_registry import ToolRegistryMixin


class PredictingAssistant(BaseAgent, ToolRegistryMixin):
    """Standalone assistant for handling predictive workflows using ComputeEngine."""
    
    def __init__(
        self,
        compute: ComputeEngine,
        ui: ChatUserInterface,
        logger: ChatLogger,
        template: Template,
        model: str = "openrouter/anthropic/claude-sonnet-4",
        temperature: float = 0.1
    ):
        """
        Initialize the predicting assistant.
        
        Args:
            compute: ComputeEngine instance for computational operations
            ui: User interface for displaying results
            logger: Logger for recording operations
            template: Jinja2 template for predictive prompts
            model: LLM model to use
            temperature: Temperature for LLM responses
        """
        # Initialize BaseAgent
        BaseAgent.__init__(self, compute, ui, logger, model, temperature)
        
        # Initialize ToolRegistryMixin
        ToolRegistryMixin.__init__(self)
        
        self.template = template
        self.icl_examples = self._load_icl_examples()
    
    def handle_predictive_query(self, query: str) -> Dict[str, Any]:
        """Execute complete predictive workflow for queries requiring ML predictions."""
        self.logger.log_info(f"Starting predictive workflow for query: {query}")
        self.ui.display_info(f"🔮 Starting predictive workflow for: {query}")
        
        # Create predictive prompt using dedicated template
        prompt = self._create_predictive_prompt(query)
        self.logger.log_debug("Generated predictive prompt")
        
        # Execute 5-step pipeline via LLM + function calls
        conversation = [{"role": "user", "content": prompt}]
        result = self._execute_predictive_workflow(conversation)
        
        self.logger.log_info("Predictive workflow completed successfully")
        self.ui.display_info("✅ Predictive workflow completed")
        
        return {
            "success": True,
            "result": result,
            "message": f"Predictive workflow completed for query: {query}"
        }
    
    def _create_predictive_prompt(self, query: str) -> str:
        """Generate prompt using predictive template."""
        metadata = self.compute.get_metadata()
        return self.template.render(
            query=query,
            schema_yaml=metadata["schema_yaml"],
            examples=self.icl_examples,
            test_timestamp=metadata["test_timestamp"].strftime("%Y-%m-%d"),
            id_csv_path=f"outputs/id_{uuid.uuid4().hex[:8]}.csv"
        )
    
    def _execute_predictive_workflow(self, conversation):
        """Execute the predictive workflow using LLM + ML tools."""
        self.logger.log_info("Starting predictive workflow execution")
        
        # Get available ML tools
        available_tools = self.tool_schemas
        self.logger.log_debug(f"Available tools: {[tool['function']['name'] for tool in available_tools]}")
        
        # Multi-turn conversation until workflow complete
        for round_idx in range(1, 10):
            self.ui.update_tool_status(f"Predictive workflow: Round {round_idx}")
            self.logger.log_debug(f"Predictive workflow round {round_idx}")
            
            # Get LLM response with streaming
            response = self.llm_client.get_response(conversation, available_tools)
            response_message = response.choices[0].message
            
            conversation.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": response_message.tool_calls if response_message.tool_calls else None
            })
            
            if not response_message.tool_calls:
                # Workflow complete
                self.logger.log_info("Predictive workflow completed - no more tool calls")
                return response_message.content
            
            # Execute ML tools
            self.logger.log_info(f"Executing {len(response_message.tool_calls)} tool call(s)")
            tool_results = self._execute_tool_calls(response_message.tool_calls)
            conversation.extend(tool_results)
        
        warning_msg = "Predictive workflow reached maximum rounds"
        self.logger.log_warning(warning_msg)
        self.ui.display_warning(warning_msg)
        return "Predictive workflow completed (max rounds reached)"
    
    def _execute_tool_calls(self, tool_calls: List) -> List[Dict]:
        """Execute tool calls and return results for conversation."""
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            
            self.ui.begin_function_execution(function_name, arguments)
            self.logger.log_function_call(function_name, arguments)
            
            start_time = time.time()
            
            tool, func, kwargs = self.parse_tool_call(tool_call)
            result = func(**kwargs)
            
            duration = time.time() - start_time
            
            self.ui.end_function_execution(result)
            self.logger.log_function_result(function_name, result.get("success", False), duration, result)
            
            # Add tool result to conversation history
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(result, indent=2)
            })
        
        return results
    
    @ToolRegistryMixin.tool
    def make_training_labels(
        self,
        target_table_name: Annotated[str, Field(description="Name of the target table (e.g., 'customer', 'product', 'users', 'posts')")],
        generation_instruction: Annotated[str, Field(description="Detailed instruction for generating training labels, describing what the model should predict")]
    ) -> Dict[str, Any]:
        """Generate training labels using CodingAssistant and save to CSV file"""
        # Generate unique filename
        csv_uuid = uuid.uuid4().hex[:8]
        output_csv_path = f"outputs/training_labels_{csv_uuid}.csv"
        
        # Create outputs directory if it doesn't exist
        Path("outputs").mkdir(exist_ok=True)
        
        # Get metadata for CodingAssistant
        metadata = self.compute.get_metadata()
        
        # Load training labels template
        template_path = Path(__file__).parent.parent / "training_labels_prompt.jinja"
        with open(template_path, 'r') as f:
            template_content = f.read()
        training_labels_template = Template(template_content)
        
        # Create CodingAssistant instance
        from .coding_assistant import CodingAssistant
        coding_assistant = CodingAssistant(
            compute=self.compute,
            ui=self.ui,
            logger=self.logger,
            template=training_labels_template,
            model=self.model,
            temperature=self.temperature
        )
        
        # Use CodingAssistant to generate training labels
        response = coding_assistant.ask_with_repl(
            prompt=generation_instruction,
            template_variables={
                'schema_yaml': metadata["schema_yaml"],
                'table_info': metadata["table_info"],
                'output_csv_path': output_csv_path,
                'examples': self.icl_examples,
                'cached_timestamps': metadata["cached_timestamps"],
                'target_table_name': target_table_name,
            }
        )
        
        # Validate that the CSV file was created
        if not Path(output_csv_path).exists():
            raise ValueError(f"Training labels CSV file was not created: {output_csv_path}")
        
        # Load and validate the generated CSV
        training_labels_df = pd.read_csv(output_csv_path)
        self.compute.validate_training_labels(training_labels_df)
        
        return {
            "success": True,
            "output_csv_path": output_csv_path,
            "message": f"Training labels generated successfully for {target_table_name} and saved to {output_csv_path}"
        }

    @ToolRegistryMixin.tool
    def fit(
        self,
        target_table_name: Annotated[str, Field(description="Name of the target table (e.g., 'customer', 'product', 'users', 'posts')")],
        task_type: Annotated[Literal["classification", "regression"], Field(description="Type of prediction task - 'classification' for binary/categorical outcomes, 'regression' for numerical outcomes")],
        training_labels_csv: Annotated[str, Field(description="Path to CSV file containing pre-generated training labels with columns ['__id', '__timestamp', '__label']")]
    ) -> Dict[str, Any]:
        """Fit a predictive model using pre-generated training labels from CSV file for a specific target table and task type"""
        self.compute.fit(
            target_table_name=target_table_name,
            task_type=task_type,
            training_labels_csv=training_labels_csv
        )
        return {
            "success": True,
            "message": f"Model fitted successfully for {target_table_name} with task type {task_type} using training labels from {training_labels_csv}"
        }
    
    @ToolRegistryMixin.tool
    def predict(
        self,
        target_table_name: Annotated[str, Field(description="Name of the target table that was used for fitting")],
        entity_ids_file: Annotated[str, Field(description="Path to CSV file containing entity IDs as a pandas Series (single column)")]
    ) -> Dict[str, Any]:
        """Make batch predictions for entities listed in a CSV file using the fitted model"""
        result_file_path = self.compute.predict_batch(
            target_table_name=target_table_name,
            test_primary_keys_file=entity_ids_file
        )
        
        # Read results for summary
        results_df = pd.read_csv(result_file_path)
        sample_rows = results_df.head(10).to_string(index=False) if len(results_df) > 0 else "No results"
        
        return {
            "success": True,
            "output_file": result_file_path,
            "num_entities": len(results_df),
            "sample_rows": sample_rows,
            "message": f"Batch predictions completed for {len(results_df)} entities. Results saved to {result_file_path}"
        }
    
    @ToolRegistryMixin.tool
    def shap(
        self,
        target_table_name: Annotated[str, Field(description="Name of the target table that was used for fitting")],
        test_primary_key_value: Annotated[str, Field(description="Primary key value of the entity to explain (e.g., customer ID, product ID)")]
    ) -> Dict[str, Any]:
        """Generate SHAP explanation for a prediction to understand feature importance"""
        shap_result = self.compute.shap(
            target_table_name=target_table_name,
            test_primary_key_value=test_primary_key_value
        )["shap_analysis"]
        
        # Clean up numpy arrays for JSON serialization
        for k, v in shap_result.items():
            if isinstance(v, np.ndarray):
                shap_result[k] = v.tolist()
        
        return {
            "success": True,
            "shap_result": shap_result,
            "message": f"SHAP explanation generated for {target_table_name} entity {test_primary_key_value}"
        }
    
    @ToolRegistryMixin.tool
    def coding_assistant(
        self,
        instruction: Annotated[str, Field(description="Instruction for data analysis, entity ID discovery, or post-processing predictions")]
    ) -> Dict[str, Any]:
        """Interactive coding assistant for data analysis and entity ID discovery with persistent REPL"""
        # Get metadata for CodingAssistant
        metadata = self.compute.get_metadata()
        
        # Load coding assistant template
        template_path = Path(__file__).parent.parent / "coding_assistant_prompt.jinja"
        with open(template_path, 'r') as f:
            template_content = f.read()
        coding_template = Template(template_content)
        
        # Create CodingAssistant instance
        from .coding_assistant import CodingAssistant
        coding_assistant = CodingAssistant(
            compute=self.compute,
            ui=self.ui,
            logger=self.logger,
            template=coding_template,
            model=self.model,
            temperature=self.temperature
        )
        
        response = coding_assistant.ask_with_repl(
            prompt=instruction,
            template_variables={
                'schema_yaml': metadata["schema_yaml"],
                'table_info': metadata["table_info"]
            }
        )
        
        return {
            "success": True,
            "response": response,
            "message": "Coding assistant completed successfully"
        }
    
    def _load_icl_examples(self) -> List[Dict[str, Any]]:
        """Load in-context learning examples for prompt rendering."""
        examples_path = Path(__file__).parent.parent / "icl_examples" / "icl_examples.json"
        with open(examples_path, 'r') as f:
            examples_data = json.load(f)
        
        # Load the actual schema and Python code for each example
        enriched_examples = []
        for example in examples_data:
            # Load schema
            schema_file = example.get("schema_file", "").replace("tabular_chat_predictor/", "")
            if schema_file:
                schema_path = Path(__file__).parent.parent / schema_file
                with open(schema_path, 'r') as f:
                    schema_yaml = f.read()
            else:
                schema_yaml = ""
            
            # Load Python code
            python_file = example.get("python_script_file", "").replace("tabular_chat_predictor/", "")
            if python_file:
                python_path = Path(__file__).parent.parent / python_file
                with open(python_path, 'r') as f:
                    python_code = f.read()
            else:
                python_code = ""
            
            enriched_examples.append({
                "schema_yaml": schema_yaml,
                "instruction": example.get("training_label_description", ""),
                "python_code": python_code
            })
        
        return enriched_examples