# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FinanceQA AI Agent project designed to improve upon the 54.1% baseline performance on the FinanceQA benchmark through multi-step reasoning, financial calculation tools, RAG systems, and document parsing capabilities.

## Common Development Commands

### Environment Setup
```bash
# Activate the conda environment
conda activate financeqa-agent

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/agent/test_core.py

# Run tests with verbose output
pytest -v

# Generate coverage report
pytest --cov=src --cov-report=html
```

### Development Tools
```bash
# Start Jupyter notebook for experimentation
jupyter notebook

# Interactive Python shell with project context
ipython
```

## Architecture Overview

### Core Components
- **Agent Core** (`src/agent/`): Multi-step reasoning orchestration
- **Model Management** (`src/models/`): Ollama ↔ OpenAI switching infrastructure
- **Financial Tools** (`src/tools/`): NPV, IRR, ratios, EBITDA calculators
- **RAG System** (`src/rag/`): ChromaDB-based knowledge retrieval
- **Document Parsing** (`src/parsers/`): 10-K and financial statement processing
- **Question Classification** (`src/classifiers/`): Tactical vs conceptual routing
- **Evaluation** (`src/evaluation/`): FinanceQA benchmark runner

### Data Organization
- `data/knowledge_base/`: Financial knowledge corpus for RAG
- `data/datasets/`: FinanceQA benchmark and evaluation datasets
- `data/sample_documents/`: Test financial documents (10-K forms, etc.)

### Configuration
- `config/`: Agent configuration and prompt templates
- `.env`: Environment variables (API keys, model settings)
- `pyproject.toml`: Tool configurations (pytest, black, mypy, flake8)

## Task Management

**Primary Task Reference**: See `tasks/prd-financeqa-ai-agent.md` for the complete Product Requirements Document.

**Implementation Tasks**: See `tasks/tasks-prd-financeqa-ai-agent.md` for detailed implementation tasks organized in 10 major phases:


## Key Design Decisions

### Model Strategy
- **Development**: Use Ollama for fast local iteration
- **Validation**: Use OpenAI models for final performance evaluation
- **Architecture**: Model-agnostic interfaces for easy switching

## Development Workflow

1. **Start with tasks**: Follow the numbered task sequence in `tasks/tasks-prd-financeqa-ai-agent.md`
2. **Test-driven development**: Write tests alongside implementation
3. **Incremental commits**: Commit after completing each major task section
4. **Documentation**: Update docstrings and comments as you code

## Entry Points

- Main agent interface: `src/agent/core.py` (to be implemented)
- CLI interface: `src/agent/cli.py` (to be implemented)
- Configuration: `config/agent_config.py` (to be implemented)
- Evaluation runner: `src/evaluation/benchmark_runner.py` (to be implemented)