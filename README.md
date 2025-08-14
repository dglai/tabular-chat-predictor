# 🔮 Tabular Chat Predictor

Interactive LLM-driven predictive analytics for tabular data using TabPFN and intelligent conversation orchestration.

## ✨ Features

- 🔮 **Make Predictions**: Train ML models and predict outcomes using TabPFN
- 📊 **Explain Results**: Use SHAP to understand why predictions were made
- 💻 **Explore Data**: Write custom code to analyze your database with persistent REPL
- 🎯 **Answer Questions**: Get insights about your data through natural language
- 🌐 **Dual Interfaces**: Both Streamlit web UI and console interface
- ⚡ **Streaming Responses**: Real-time LLM response streaming
- 🧠 **Intelligent Orchestration**: Automatic tool selection and workflow management

## 🚀 Quick Start

### Installation

**Recommended: Using Conda Environment**

```bash
# Create a new conda environment
conda create -n tabular-chat-predictor python=3.10
conda activate tabular-chat-predictor

# Clone or download the repository
cd tabular-chat-predictor

# Install the package
pip install .
```

### Basic Usage

#### Streamlit Web Interface (Recommended)

```bash
# Launch the web interface
streamlit run streamlit_chat_predictor_new.py
```

#### Console Interface

```bash
# Run with Amazon dataset
python -m tabular_chat_predictor.frontends.console_frontend datasets/demo/rel-amazon-input "2016-01-01"

# Run with StackExchange dataset  
python -m tabular_chat_predictor.frontends.console_frontend datasets/demo/rel-stack-input "2021-01-01"
```

Or use the installed command:

```bash
tabular-chat-predictor datasets/demo/rel-amazon-input "2016-01-01"
```

## 📋 Requirements

### Core Dependencies

- **Python**: 3.8+
- **TabPFN**: For fast neural network predictions on tabular data
- **LiteLLM**: For LLM integration with multiple providers
- **Streamlit**: For the web interface
- **Rich**: For beautiful console output

### Complete Dependencies

See [`requirements.txt`](requirements.txt) for the full list of dependencies (~15-20 packages).

## 🗂️ Dataset Format

The predictor works with datasets in DFS (Deep Feature Synthesis) format:

```
datasets/demo/rel-amazon-input/
├── __dfs__/
│   ├── customer.npz    # Customer features
│   └── product.npz     # Product features  
└── metadata.yaml       # Schema definition
```

### Sample Datasets Included

- **Amazon E-commerce**: Customer and product data for churn/LTV prediction
- **StackExchange**: User and post data for engagement prediction

## 💬 Example Queries

### Amazon Dataset
- "Would user ID 1592898 buy or review any product in the next quarter?"
- "How much will user ID 1593600 spend in the next quarter?"
- "Would product ID 420601 be bought or reviewed by any user in the next quarter?"

### StackExchange Dataset  
- "Would user ID 2666 make any votes, posts or comments in the next quarter?"
- "Would user ID 76999 receive any new badge in the next quarter?"
- "How many votes would post ID 115603 receive in the next quarter?"

## 🏗️ Architecture

### Protocol-Based Design
- **Frontend-agnostic orchestrator** that works with any UI
- **Protocol-based interfaces** for clean separation of concerns
- **Unified function execution** with consistent error handling

### Core Components

```
tabular_chat_predictor/
├── agents/           # LLM agents (orchestrator, coding, predicting)
├── compute/          # Core computation engine and TabPFN integration  
├── core/             # Protocols, tools, and base classes
├── frontends/        # Streamlit and console frontends
├── interfaces/       # UI and logging implementations
├── templates/        # Jinja2 prompt templates
└── icl_examples/     # In-context learning examples
```

## 🔧 Configuration

### Environment Variables

```bash
# Set your LLM model (optional, defaults to Claude)
export LITELLM_MODEL="openai/gpt-4"

# Set API keys as needed
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### Model Options

The predictor supports any model compatible with LiteLLM:
- `openai/gpt-4`
- `anthropic/claude-3-5-sonnet-20241022`
- `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`
- And many more...

## 📖 Usage Examples

### 1. Basic Prediction Workflow

```python
from tabular_chat_predictor.frontends.console_frontend import main
from tabular_chat_predictor.compute.compute_engine import ComputeEngine
from datetime import datetime

# Initialize compute engine
compute = ComputeEngine(
    dataset_path="datasets/demo/rel-amazon-input",
    test_timestamp=datetime(2016, 1, 1)
)

# The orchestrator handles the rest automatically
```

### 2. Custom Data Analysis

Ask questions like:
- "Show me the distribution of customer ages"
- "What are the top 10 products by sales?"
- "Create a visualization of user engagement over time"

### 3. Predictive Analytics

Ask questions like:
- "Which customers are most likely to churn?"
- "What will be the lifetime value of user X?"
- "Explain why this prediction was made"

## 🛠️ Development

### Project Structure

```
tabular-chat-predictor/
├── tabular_chat_predictor/    # Main package
├── datasets/                  # Sample datasets
├── examples/                  # Usage examples
├── README.md                  # This file
├── pyproject.toml            # Package configuration
├── requirements.txt          # Dependencies
└── .gitignore               # Git ignore rules
```

### Adding New Datasets

1. Create dataset directory with `__dfs__/` subdirectory
2. Add `.npz` files for each table
3. Create `metadata.yaml` with schema definition
4. Update frontend configurations if needed

### Extending Functionality

The modular architecture makes it easy to:
- Add new LLM agents
- Create custom tools
- Implement new frontends
- Add data processing capabilities

## 🐛 Troubleshooting

### Common Issues

**Installation Problems**
```bash
# Recommended: Use conda environment
conda create -n tabular-chat-predictor python=3.9
conda activate tabular-chat-predictor
pip install .
```

**Missing Dependencies**
```bash
# Install all dependencies explicitly
pip install -r requirements.txt
```

**Virtual Environment Issues**
```bash
# If using conda (recommended)
conda create -n tabular-chat-predictor python=3.9
conda activate tabular-chat-predictor

# If using venv
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

**Dataset Loading Issues**
- Ensure dataset follows the DFS format
- Check that `metadata.yaml` is properly formatted
- Verify `.npz` files are not corrupted

**LLM Connection Issues**
- Check your API keys are set correctly
- Verify the model name is supported by LiteLLM
- Try a different model if one is not working

### Getting Help

1. Check the console output for detailed error messages
2. Enable debug mode in Streamlit interface
3. Review the logs for function execution details

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 🙏 Acknowledgments

- Built on top of [TabPFN](https://github.com/automl/TabPFN) for fast tabular predictions
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider LLM integration
- Powered by [Streamlit](https://streamlit.io/) for the web interface