# FinanceQA AI Agent

A unified multi-step reasoning AI agent designed to improve upon the 54.1% baseline performance on the FinanceQA benchmark through tool-based architecture, RAG-enhanced context retrieval, and financial calculation capabilities.

## Overview

This agent tackles complex financial analysis tasks that traditional LLMs struggle with by:
- **RAG-Enhanced Architecture**: Kaggle Financial Q&A dataset provides reasoning patterns and examples
- **3-Step Reasoning Chain**: Analysis → RAG Search → Synthesis for focused question answering
- **Enhanced Context Retrieval**: Searches financial Q&A pairs for similar reasoning examples
- **Reasoning Chain Transparency**: Complete visibility into analysis, context retrieval, and synthesis steps

## Architecture Overview

### Core Module Structure

```
src/
├── agent/                      # Agent orchestration and reasoning
│   ├── core.py                # Base reasoning chain infrastructure
│   ├── financial_agent.py     # Main FinancialAgent with RAG and tool integration.
│   └── prompts.py            # Prompt templates
│
├── models/                     # Model management and providers
│   ├── model_manager.py       # Model abstraction and switching infrastructure
│   └── ollama_provider.py     # Local Ollama integration with async support
│
├── rag/                        # RAG system and knowledge base
│   ├── financial_rag.py       # ChromaDB-based financial pattern retrieval
│   ├── data_processor.py      # Data processing for RAG ingestion
│   ├── data_ingestion.py      # Data ingestion utilities
│   └── ingest_data.py         # Simple script running one-time data ingestion
│
├── evaluation/                 # Benchmark and evaluation system
│   ├── benchmark_runner.py    # Orchestrator for complete evaluation workflow
│   ├── run_agent.py          # Agent evaluation and result generation
│   ├── run_llm_evaluation.py # LLM-based evaluation of agent results
│   └── llm_scorer.py         # LLM scoring utilities
│
└── utils/                      # Shared utilities
    └── logger.py              # Logging configuration
```

### Key Responsibilities by Module

- **`agent/`**: Core reasoning orchestration, tool coordination, and financial domain logic
- **`models/`**: Model abstraction enabling seamless switching between Ollama and OpenAI
- **`rag/`**: Knowledge retrieval system with financial patterns and Q&A ingestion
- **`evaluation/`**: Complete benchmark pipeline with agent evaluation and LLM scoring
- **`utils/`**: Shared infrastructure and logging

## Quick Start

### Prerequisites

- Python 3.9+
- Conda (recommended) or pip
- Ollama (for local models) or OpenAI API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FinanceArenaQA
   ```

2. **Create and activate environment**:
   ```bash
   conda env create -f environment.yml
   conda activate financeqa-agent
   ```

3. **Set up data and dependencies**:
   ```bash
   # One-time setup: Download datasets and setup RAG knowledge base
   ./scripts/setup_data.sh

   # Optional: Force clean start
   ./scripts/setup_data.sh --force-clean

   # Optional: Skip downloads if data exists
   ./scripts/setup_data.sh --skip-download
   ```

4. **Set up Ollama** (recommended for development):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve

   # Pull required models
   ollama pull llama2
   ollama pull phi3:mini  # Lighter model for development
   ```

5. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your model configurations
   ```

### Basic Usage

```python
import asyncio
from src.agent.financial_agent import FinancialAgent
from src.models.model_manager import create_model_manager_from_config

# Configure model manager with RAG settings
config = {
    "providers": {
        "ollama": {
            "type": "ollama",
            "model_name": "llama2",
            "base_url": "http://localhost:11434",
            "temperature": 0.1
        }
    },
    "default_provider": "ollama",
    "rag": {
        "enabled": true,
        "chroma_persist_dir": "./chroma_db",
        "collection_name": "financial_patterns",
        "confidence_thresholds": {
            "rag_min_confidence": 0.6,
            "high_confidence": 0.8
        }
    }
}

async def main():
    # Use context manager for automatic resource cleanup
    async with create_model_manager_from_config(config) as model_manager:
        # Create unified financial agent with RAG and calculation tools
        agent = FinancialAgent(model_manager, config=config, enable_rag=True)

        # Ask a financial question
        question = "Calculate the ROI if I invest $1000 and gain $1200"

        # Get reasoning chain (can use sync or async)
        chain = agent.answer_question(question)  # Sync wrapper
        # Or: chain = await agent.answer_question_async(question)  # Async

        # Print results
        print(f"Question: {question}")
        print(f"Answer: {chain.final_answer}")
        print(f"Confidence: {chain.confidence_score}")
        print(f"Steps: {len(chain.steps)}")
        print(f"RAG enabled: {agent.enable_rag}")
    # Automatic cleanup happens here

# Run the example
asyncio.run(main())
```

### Without RAG (Simple Mode)

```python
# Disable RAG for pure calculation-focused usage
agent = FinancialAgent(model_manager, config=config, enable_rag=False)
```

## End-to-End Testing

### FinanceQA Benchmark Evaluation

Evaluate agent performance on the FinanceQA benchmark. The evaluation system has three main scripts:

#### Complete Evaluation (Orchestrator)

Run both agent evaluation and LLM scoring in one command:

```bash
# Full dataset evaluation with RAG and LLM scoring
python src/evaluation/benchmark_runner.py \
    --use-llm-evaluation \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Quick evaluation (50 samples) with LLM scoring
python src/evaluation/benchmark_runner.py \
    --num-samples 50 \
    --use-llm-evaluation \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Evaluate specific question types
python src/evaluation/benchmark_runner.py \
    --question-type "assumption" \
    --num-samples 25 \
    --use-llm-evaluation \
    --dataset-path data/datasets/financeqa \
    --output-dir results/
```

#### Step-by-Step Evaluation

For more control, run agent evaluation and LLM scoring separately:

**1. Agent Evaluation (Generate Responses)**

```bash
# Full dataset with RAG (default)
python src/evaluation/run_agent.py \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Quick test (10 samples)
python src/evaluation/run_agent.py \
    --num-samples 10 \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Specific question types
python src/evaluation/run_agent.py \
    --question-type "assumption" \
    --num-samples 50 \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Disable RAG for pure calculation mode
python src/evaluation/run_agent.py \
    --disable-rag \
    --question-type "conceptual" \
    --num-samples 25 \
    --dataset-path data/datasets/financeqa \
    --output-dir results/
```

**2. LLM Evaluation (Score Results)**

```bash
# Evaluate agent results with LLM scoring
python src/evaluation/run_llm_evaluation.py \
    --result-file results/financeqa_agent_results_assumption_50_*.json \
    --output-dir results/

# Evaluate all result files matching pattern
find results/ -name "financeqa_agent_results_*.json" | \
    xargs -I {} python src/evaluation/run_llm_evaluation.py \
    --result-file {} --output-dir results/
```


## Data Flow



### 3-Step Reasoning Process

The agent uses a simplified 3-step reasoning chain for all financial questions:

#### **Step 1: Analysis**
- **Enhanced Question Analysis**: Parses question to identify financial concepts and required context
- **Source Routing Logic**: Determines optimal data sources based on question type:
  - `["10k"]`: Company-specific data questions (e.g., "What is Costco's revenue?")
  - `["qa_dataset"]`: Conceptual/definitional questions (e.g., "What is EBITDA?")
  - `["both"]`: Analysis requiring both company data and financial methodology
- **Search Strategy**: Generates optimized keywords for RAG retrieval

#### **Step 2: RAG Search**
- **Context Enhancement**: Searches Kaggle Financial Q&A dataset for similar questions and reasoning patterns
- **Reasoning Pattern Extraction**: Retrieves question-answer pairs with reasoning explanations
- **Smart Fallback**: Continues with empty context if RAG search fails or returns no results

#### **Step 3: Synthesis**
- **Answer Generation**: Combines analysis insights with RAG reasoning patterns
- **Context Integration**: Uses both provided context (if any) and retrieved reasoning examples
- **Transparency**: Provides complete reasoning chain with step-by-step decision tracking

### Reasoning Chain Inspection

Every question generates a complete 3-step reasoning chain with full transparency:

```python
# Inspect reasoning process (3 steps: Analysis → RAG → Synthesis)
for step in chain.steps:
    print(f"Step: {step.description}")
    print(f"Status: {step.status.value}")
    print(f"Type: {step.step_type.value}")  # analysis, context_rag, synthesis
    print(f"Execution Time: {step.execution_time_ms}ms")
    if step.error_message:
        print(f"Error: {step.error_message}")
    print("---")

# Access specific step outputs
analysis_step = chain.get_step_by_type(StepType.ANALYSIS)
rag_step = chain.get_step_by_type(StepType.CONTEXT_RAG)
synthesis_step = chain.get_step_by_type(StepType.SYNTHESIS)

print(f"Final Answer: {chain.final_answer}")
print(f"Total Time: {chain.total_execution_time_ms}ms")
```

## Configuration

### Configuration Files

```
config/                        # Configuration templates and patterns
pyproject.toml                # Project metadata and tool configurations
.env                          # Environment variables (API keys, model settings)
environment.yml               # Conda environment specification
scripts/setup_data.sh         # One-time data setup script
```

### Model Configuration

Edit `config/model_config.json`:

```json
{
  "providers": {
    "ollama": {
      "type": "ollama",
      "model_name": "llama2",
      "base_url": "http://localhost:11434",
      "temperature": 0.1,
      "timeout": 60,
      "options": {
        "num_predict": 512,
        "top_p": 0.9,
        "repeat_penalty": 1.1
      }
    },
    "openai": {
      "type": "openai",
      "model_name": "gpt-4",
      "api_key": "your-api-key",
      "temperature": 0.1
    }
  },
  "default_provider": "ollama",
  "rag": {
    "enabled": true,
    "chroma_persist_dir": "./chroma_db",
    "collection_name": "financial_patterns",
    "config_path": "config/context_patterns.yaml",
    "confidence_thresholds": {
      "rag_min_confidence": 0.6,
      "high_confidence": 0.8
    },
    "retrieval_settings": {
      "max_results": 10,
      "similarity_threshold": 0.7
    },
    "fallback_behavior": {
      "continue_without_rag": true,
      "log_failures": true
    }
  }
}
```

### RAG Configuration Options

- `enabled`: Enable/disable RAG functionality
- `chroma_persist_dir`: Directory for ChromaDB persistence
- `collection_name`: ChromaDB collection for financial patterns
- `confidence_thresholds`: Minimum confidence levels for using RAG results
- `retrieval_settings`: Control number and quality of retrieved results
- `fallback_behavior`: How to handle RAG failures

### Agent Configuration

Key configuration parameters:

- `temperature`: Model creativity (0.1 for financial precision)
- `enable_rag`: Enable/disable RAG functionality (default: True)
- `rag.chroma_persist_dir`: ChromaDB storage directory
- `rag.collection_name`: Collection name for financial knowledge
- `rag.max_results`: Maximum RAG search results per query



## Troubleshooting

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
chain = await agent.answer_question(question)

# Inspect failed steps
failed_steps = chain.get_failed_steps()
for step in failed_steps:
    print(f"Failed: {step.description} - {step.error_message}")
```