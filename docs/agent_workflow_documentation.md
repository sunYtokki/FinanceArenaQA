# FinanceQA Agent Workflow Documentation

## Overview

This document provides a comprehensive overview of the FinanceQA AI Agent workflow, based on careful analysis of the current implementation in the codebase. All information has been cross-referenced with the actual code to ensure accuracy.

## Architecture Components

### 1. Core Agent Structure

**Primary Entry Point**: `src/agent/financial_agent.py:FinancialAgent`

The FinancialAgent implements a simplified 3-step reasoning chain:

```
Question Input → Analysis → RAG Search → Synthesis → Final Answer
```

**Key Implementation Points** (verified in codebase):
- **Initialization** (`__init__`, lines 17-45): Sets up model manager, RAG system, and configuration
- **Resource Management** (`cleanup()`, lines 47-68): Handles proper cleanup of model connections
- **Main Entry** (`answer_question()`, lines 123-175): Orchestrates the complete reasoning process

### 2. Reasoning Chain Architecture

**Core Module**: `src/agent/core.py`

The reasoning system uses a structured approach with the following components:

#### StepType Enumeration (lines 23-29):
- `ANALYSIS`: Question analysis and requirements identification
- `CONTEXT_RAG`: RAG-based context retrieval
- `CALCULATION`: Financial calculations (defined but not currently used)
- `VALIDATION`: Answer validation (defined but not currently used)
- `SYNTHESIS`: Final answer generation

#### ReasoningChain Class (lines 66-113):
- **Unique ID**: Each reasoning chain has a UUID for tracking
- **Step Management**: Maintains ordered list of reasoning steps
- **Final Answer**: Stores the complete answer string
- **Timing**: Tracks total execution time in milliseconds
- **Serialization**: Converts to dictionary format for storage

## Detailed Workflow Steps

### Step 1: Enhanced Analysis

**Implementation**: `src/agent/financial_agent.py:_execute_analysis()` (lines 178-251)

**Purpose**: Analyze the financial question to identify required concepts and route to appropriate data sources.

**Process Flow**:
1. **Prompt Generation** (line 194): Uses `get_enhanced_analysis_prompt()` from `src/agent/prompts.py`
2. **LLM Analysis** (lines 196-199): Calls model manager to analyze the question
3. **JSON Parsing** (lines 202-230): Extracts structured analysis output
4. **Error Handling** (lines 231-245): Fallback for parsing failures

**Output Data Structure**:
```json
{
  "financial_concepts": ["EBITDA", "earnings", "operating performance"],
  "required_context": ["definition", "calculation methodology"],
  "search_keywords": ["EBITDA definition", "earnings before interest"],
  "recommended_sources": ["qa_dataset"]
}
```

**Source Routing Logic** (verified in `src/agent/prompts.py:20-34`):
- **["10k"]**: Company-specific data requests (e.g., "What is Costco's revenue?")
- **["qa_dataset"]**: General financial concepts (e.g., "What is EBITDA?")
- **["both"]**: Questions requiring both theory and company data

### Step 2: RAG Context Enhancement

**Implementation**: `src/agent/financial_agent.py:_enhance_context()` (lines 253-369)

**Purpose**: Search knowledge base for similar questions and reasoning patterns.

**Process Flow**:
1. **Search Query Preparation** (lines 284-303): Combines question with analysis-derived keywords
2. **RAG Search Execution** (line 306): Calls `FinancialRAG.search()` with source filtering
3. **Context Building** (lines 314-333): Extracts reasoning patterns from similar Q&A pairs
4. **Result Packaging** (lines 334-357): Structures context data for synthesis step

**Key Features**:
- **Enhanced Search**: Uses multiple keywords from analysis step
- **Source Filtering**: Routes to specific data sources based on analysis
- **Reasoning Context**: Extracts step-by-step reasoning from similar questions

### Step 3: Synthesis and Answer Generation

**Implementation**: `src/agent/financial_agent.py:_synthesis_answer()` (lines 409-492)

**Purpose**: Generate final answer using analysis insights and RAG context.

**Process Flow**:
1. **Data Aggregation** (lines 425-435): Combines analysis and RAG results
2. **Prompt Construction** (lines 437-439): Uses `get_synthesis_prompt()`
3. **Answer Generation** (lines 441-444): LLM generates complete response
4. **Answer Extraction** (lines 446-460): Parses final answer from response
5. **Transparency Building** (lines 463-479): Creates reasoning transparency record

## RAG System Architecture

**Primary Module**: `src/rag/financial_rag.py:FinancialRAG`

### ChromaDB Integration

**Initialization** (lines 29-51):
- **Persistent Client**: Uses ChromaDB with local persistence
- **Unified Collection**: Single collection `financial_knowledge` with metadata filtering
- **Source Types**: Supports "10k" and "qa_dataset" source filtering

### Search Functionality

**Main Search Method** (`search()`, lines 57-131):
```python
def search(
    question: Union[str, List[str]],
    n_results: int = 5,
    source_filter: Optional[Union[str, List[str]]] = None
) -> List[Dict[str, Any]]
```

**Specialized Search Methods**:
- `search_10k_only()`: Filters to 10-K documents only
- `search_qa_only()`: Filters to Q&A dataset only
- `search_all_sources()`: No filtering applied

### Document Management

**Document Addition** (`add_documents()`, lines 169-226):
- **Metadata Validation**: Requires `source_type` field
- **Deduplication**: Skips existing documents by default
- **Batch Processing**: Handles multiple documents efficiently

**Collection Statistics** (`get_collection_stats()`, lines 267-303):
- **Total Count**: Overall document count
- **Source Breakdown**: Count by source type
- **Availability Check**: Verifies ChromaDB connection

## Evaluation Pipeline

### Agent Result Generation

**Module**: `src/evaluation/run_agent.py`

**Key Components**:
- **EvaluationExample** (lines 25-48): Data structure for dataset entries
- **AgentResponse** (lines 52-58): Structured agent output
- **Dataset Loading** (lines 71-94): JSONL file processing

### LLM-Based Evaluation

**Module**: `src/evaluation/run_llm_evaluation.py`

**Process**:
1. **Result File Loading**: Processes agent-generated results
2. **LLM Scoring**: Uses separate LLM to evaluate answer quality
3. **Metrics Calculation**: Computes accuracy and error rates

### Orchestration

**Module**: `src/evaluation/benchmark_runner.py:BenchmarkOrchestrator`

**Workflow** (`run_evaluation()`, lines 75-172):
1. **Agent Evaluation**: Generates initial results using agent
2. **Optional LLM Evaluation**: Adds LLM-based scoring if requested
3. **Result Saving**: Persists final evaluation metrics
4. **Summary Generation**: Creates comprehensive evaluation report

## Model Management

**Module**: `src/models/model_manager.py`

**Architecture**:
- **Provider Abstraction**: `ModelProvider` base class for different LLM providers
- **Response Structure**: `ModelResponse` with content, model info, and metadata
- **Configuration**: `ModelConfig` with temperature, timeouts, and provider options

**Key Features**:
- **Multi-Provider Support**: Can switch between Ollama, OpenAI, etc.
- **Synchronous Wrapper**: `generate_sync()` method for non-async contexts
- **Configuration Validation**: Ensures valid provider settings

## Prompt Engineering

**Module**: `src/agent/prompts.py`

### Analysis Prompt (lines 5-95)
- **Structured Output**: JSON format with specific fields
- **Source Routing**: Logic for data source selection
- **Example-Driven**: Clear examples for different question types

### Context Builder Prompt (lines 99-137)
- **Pattern Extraction**: Identifies reasoning patterns from similar questions
- **Data Mapping**: Links reasoning steps to available context
- **Solution Framework**: Structured approach to problem-solving

### Synthesis Prompt (lines 142-163)
- **Direct Answer Focus**: Emphasizes specific, accurate responses
- **Step-by-Step Calculation**: Supports mathematical reasoning
- **Final Answer Extraction**: Clear format for answer parsing

## Data Flow Summary

```
Input Question
     ↓
[Step 1: Analysis]
- Parse question for concepts
- Determine required context
- Route to data sources
     ↓
[Step 2: RAG Search]
- Query knowledge base
- Filter by source type
- Extract reasoning patterns
     ↓
[Step 3: Synthesis]
- Combine analysis + context
- Generate final answer
- Extract and format result
     ↓
Final Answer + Reasoning Chain
```

## Configuration and Setup

**Environment Variables**: Managed through `.env` file
**Model Configuration**: JSON-based configuration in `config/` directory
**Dataset Path**: Configurable path to FinanceQA dataset
**Output Directory**: Configurable results storage location

## Error Handling and Robustness

**Graceful Degradation**:
- RAG system unavailable → empty context with warning
- LLM parsing errors → structured error responses
- Model generation failures → fallback messages

**Resource Management**:
- Explicit cleanup methods for model connections
- Proper exception handling in critical paths
- Timeout configuration for model calls
