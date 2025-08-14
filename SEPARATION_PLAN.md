# Tab2Graph Webdemo → Tabular Chat Predictor
## Separation Plan & Implementation Guide

### 🎯 Project Overview

**Objective**: Separate the Tab2Graph webdemo into a clean, standalone "tabular-chat-predictor" repository that maintains all current functionality while being completely independent from the main tab2graph codebase.

**Target Repository**: `tabular-chat-predictor`
**Current Source**: `tab2graph/webdemo/`

---

## 📋 Architecture Analysis

### Current Webdemo Structure
```
tab2graph/webdemo/
├── __init__.py                    # Package initialization
├── agents/                        # LLM agents and orchestration
│   ├── base_agent.py             # Base agent functionality
│   ├── coding_assistant.py       # Interactive coding with REPL
│   ├── llm_client.py             # LiteLLM integration
│   ├── orchestrator.py           # Main conversation orchestrator
│   └── predicting_assistant.py   # ML prediction workflows
├── compute/                       # Core computational engine
│   ├── compute_engine.py         # Main computational interface
│   ├── dataset_loader.py         # DFS dataset loading
│   ├── repl_manager.py           # IPython REPL management
│   ├── tabpfn_manager.py         # TabPFN model management
│   └── tabpfn_predictor.py       # TabPFN prediction interface
├── core/                          # Core protocols and tools
│   ├── protocols.py              # Abstract interfaces
│   ├── tool.py                   # Tool creation utilities
│   └── tool_registry.py         # Tool registration system
├── frontends/                     # User interfaces
│   ├── console_frontend.py       # Command-line interface
│   └── streamlit_frontend.py     # Web-based interface
├── interfaces/                    # Interface implementations
│   ├── console_interface.py      # Console UI implementation
│   └── streamlit_interface.py    # Streamlit UI implementation
├── templates/                     # Jinja2 prompt templates
│   ├── orchestrator_prompt.jinja
│   └── predictive_query_prompt.jinja
├── icl_examples/                  # In-context learning examples
│   ├── amazon/                   # Amazon dataset examples
│   └── stack/                    # StackExchange examples
├── coding_assistant_prompt.jinja # Coding assistant template
├── schema_to_mermaid.py          # Schema visualization
└── training_labels_prompt.jinja  # Training label generation
```

### Key Features to Preserve
- ✅ **ML Predictions**: Complete TabPFN-based prediction pipeline
- ✅ **SHAP Explanations**: Model interpretability with SHAP
- ✅ **Data Exploration**: Interactive coding assistant with persistent REPL
- ✅ **Dual Interfaces**: Both Streamlit web UI and console interface
- ✅ **LLM Orchestration**: Intelligent tool selection and workflow management
- ✅ **Streaming Responses**: Real-time LLM response streaming

### Dependencies Analysis
**Current Dependencies** (from webdemo code analysis):
- **Core ML**: `tabpfn`, `tabpfn-extensions`, `shap`, `scikit-learn`
- **LLM Integration**: `litellm`, `jinja2`, `pydantic`
- **UI Components**: `streamlit`, `streamlit-mermaid`, `rich`, `prompt-toolkit`
- **Data Processing**: `pandas`, `numpy`, `duckdb`, `yaml`
- **Python Environment**: `ipython`, `matplotlib`, `seaborn`
- **Standard Libraries**: `pathlib`, `datetime`, `json`, `uuid`, `logging`

**Estimated Clean Dependencies**: ~15-20 packages (vs current 250+)

---

## 🏗️ Target Repository Structure

```
tabular-chat-predictor/
├── tabular_chat_predictor/        # Main package (no src/ needed)
│   ├── __init__.py
│   ├── agents/                    # LLM agents (orchestrator, coding, predicting)
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── coding_assistant.py
│   │   ├── llm_client.py
│   │   ├── orchestrator.py
│   │   └── predicting_assistant.py
│   ├── compute/                   # Core computation engine and TabPFN integration
│   │   ├── __init__.py
│   │   ├── compute_engine.py
│   │   ├── dataset_loader.py
│   │   ├── repl_manager.py
│   │   ├── tabpfn_manager.py
│   │   └── tabpfn_predictor.py
│   ├── core/                      # Protocols, tools, and base classes
│   │   ├── __init__.py
│   │   ├── protocols.py
│   │   ├── tool.py
│   │   └── tool_registry.py
│   ├── frontends/                 # Streamlit and console frontends
│   │   ├── __init__.py
│   │   ├── console_frontend.py
│   │   └── streamlit_frontend.py
│   ├── interfaces/                # UI and logging implementations
│   │   ├── __init__.py
│   │   ├── console_interface.py
│   │   └── streamlit_interface.py
│   ├── templates/                 # Jinja2 prompt templates
│   │   ├── orchestrator_prompt.jinja
│   │   └── predictive_query_prompt.jinja
│   ├── icl_examples/              # In-context learning examples
│   │   ├── icl_examples.json
│   │   ├── amazon/
│   │   └── stack/
│   ├── coding_assistant_prompt.jinja
│   ├── schema_to_mermaid.py
│   └── training_labels_prompt.jinja
├── datasets/                      # Sample datasets (Amazon, StackExchange)
│   ├── demo/
│   │   ├── rel-amazon-input/
│   │   │   ├── __dfs__/
│   │   │   └── metadata.yaml
│   │   └── rel-stack-input/
│   │       ├── __dfs__/
│   │       └── metadata.yaml
├── examples/                      # Usage examples and scripts
│   ├── basic_usage.py
│   ├── streamlit_demo.py
│   └── console_demo.py
├── README.md                      # Comprehensive setup and usage guide
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Clean dependency list
└── .gitignore                     # Git ignore file
```

---

## 📝 Implementation Plan (10 Steps)

### Step 1: Analyze Dependencies and Create Clean List
**Status**: Pending
**Tasks**:
- Extract essential dependencies from current requirements.txt (250+ → ~15-20)
- Identify webdemo-specific packages vs tab2graph-specific packages
- Create minimal requirements.txt with only necessary packages

**Key Dependencies**:
```
# Core ML
tabpfn>=2.0.9
tabpfn-extensions>=0.0.4
shap>=0.48.0
scikit-learn>=1.6.0

# LLM Integration  
litellm>=1.74.0
jinja2>=3.1.6
pydantic>=2.11.7

# UI Components
streamlit>=1.46.0
streamlit-mermaid>=0.3.0
rich>=14.0.0
prompt-toolkit

# Data Processing
pandas>=2.2.3
numpy>=2.2.0
duckdb>=1.3.0
pyyaml>=6.0.2

# Python Environment
ipython
matplotlib>=3.10.0
seaborn>=0.12.0
```

### Step 2: Design New Repository Structure
**Status**: Pending
**Tasks**:
- Create directory structure for `tabular-chat-predictor`
- Plan package layout under `tabular_chat_predictor/` (no src/ needed)
- Design datasets and examples directories

### Step 3: Create Project Configuration Files
**Status**: Pending
**Tasks**:
- Create `pyproject.toml` with package metadata and dependencies
- Create clean `requirements.txt`
- Create `.gitignore` file
- Set up package entry points for CLI usage

**pyproject.toml Structure**:
```toml
[project]
name = "tabular-chat-predictor"
version = "0.1.0"
description = "Interactive LLM-driven predictive analytics for tabular data"
dependencies = [
    # Clean dependency list from Step 1
]

[project.scripts]
tabular-chat-predictor = "tabular_chat_predictor.frontends.console_frontend:main"
```

### Step 4: Copy and Refactor Core Code
**Status**: Pending
**Tasks**:
- Copy all files from `tab2graph/webdemo/` to new structure
- Remove any tab2graph-specific code or references
- Ensure all functionality is preserved

**Files to Copy**:
- All `.py` files from webdemo directory
- All `.jinja` template files
- All ICL examples and data

### Step 5: Update Import Statements
**Status**: Pending
**Tasks**:
- Replace all `from tab2graph.webdemo.*` imports with `from tabular_chat_predictor.*`
- Update relative imports to match new package structure
- Ensure no remaining references to tab2graph package

**Import Changes**:
```python
# Before
from tab2graph.webdemo.compute.compute_engine import ComputeEngine
from tab2graph.webdemo.agents.predicting_assistant import PredictingAssistant

# After  
from tabular_chat_predictor.compute.compute_engine import ComputeEngine
from tabular_chat_predictor.agents.predicting_assistant import PredictingAssistant
```

### Step 6: Create Sample Datasets Directory
**Status**: Pending
**Tasks**:
- Copy Amazon and StackExchange sample datasets
- Ensure datasets are in correct DFS format with `__dfs__/` directories
- Include metadata.yaml files for each dataset
- Verify datasets work with existing data loaders

**Dataset Structure**:
```
datasets/demo/
├── rel-amazon-input/
│   ├── __dfs__/
│   │   ├── customer.npz
│   │   └── product.npz
│   └── metadata.yaml
└── rel-stack-input/
    ├── __dfs__/
    │   ├── users.npz
    │   └── posts.npz
    └── metadata.yaml
```

### Step 7: Update Configuration and Template Files
**Status**: Pending
**Tasks**:
- Update dataset paths in frontend configurations
- Modify Jinja2 templates to remove tab2graph references
- Update example queries and ICL examples
- Ensure all hardcoded paths are relative to new structure

**Configuration Updates**:
- Update dataset paths in `streamlit_frontend.py`
- Modify template loading paths in agents
- Update ICL example file paths

### Step 8: Create Comprehensive README
**Status**: Pending
**Tasks**:
- Write installation instructions
- Document usage examples for both interfaces
- Explain dataset format requirements
- Provide troubleshooting guide

**README Sections**:
```markdown
# Tabular Chat Predictor

## Installation
## Quick Start
## Features
## Usage Examples
## Dataset Format
## Configuration
## Troubleshooting
```

### Step 9: Add Example Scripts
**Status**: Pending
**Tasks**:
- Create `examples/basic_usage.py` - Simple prediction example
- Create `examples/streamlit_demo.py` - Streamlit interface demo
- Create `examples/console_demo.py` - Console interface demo
- Add Jupyter notebook examples

### Step 10: Validate Project Works
**Status**: Pending
**Tasks**:
- Test that the project can be launched with a single command
- Verify basic functionality works end-to-end

**Validation Checklist**:
- [ ] Project installs successfully with `pip install -e .`
- [ ] Streamlit interface launches with single command
- [ ] Console interface launches with single command
- [ ] Basic prediction workflow completes successfully

---

## 🚀 Expected Benefits

### For Users
- **Easy Installation**: Simple `pip install tabular-chat-predictor`
- **Self-Contained**: No dependency on large tab2graph framework
- **Immediate Usability**: Working examples and sample datasets included
- **Clear Documentation**: Comprehensive setup and usage guide

### For Developers
- **Clean Codebase**: Focused, minimal dependencies
- **Maintainable**: Simple structure easy to understand and modify
- **Extensible**: Well-architected for adding new features
- **Independent**: No external framework dependencies

### Technical Advantages
- **Minimal Dependencies**: ~15-20 packages vs 250+ in original
- **Fast Installation**: Quick pip install without heavy dependencies
- **Portable**: Easy to deploy in different environments
- **Focused**: Purpose-built for tabular data chat prediction

---

## 🎯 Success Criteria

The separation will be considered successful when:

1. **Complete Independence**: No imports from tab2graph package
2. **Full Functionality**: All current features work identically
3. **Easy Setup**: One-command installation and startup
4. **Working Examples**: All provided examples run successfully
5. **Sample Data**: Included datasets work out of the box
6. **Clear Documentation**: Users can get started quickly from README

---

## 📅 Implementation Timeline

**Estimated Time**: 1-2 days for core implementation
**Priority Order**: Steps 1-5 are critical path, Steps 6-10 can be done in parallel

This plan maintains the excellent architecture of the current webdemo while making it completely standalone and accessible to users who want a powerful tabular data chat predictor without the full tab2graph framework.