# Tasks: FinanceQA AI Agent

## Relevant Files

- `src/agent/core.py` - Main agent orchestration and reasoning chain implementation
- `src/agent/core.test.py` - Unit tests for core agent functionality
- `src/models/model_manager.py` - Model switching infrastructure (Ollama ↔ OpenAI)
- `src/models/model_manager.test.py` - Unit tests for model management
- `src/tools/financial_calculator.py` - Financial calculation tools (NPV, IRR, ratios, EBITDA)
- `src/tools/financial_calculator.test.py` - Unit tests for financial calculations
- `src/tools/code_executor.py` - Secure Python code execution environment
- `src/tools/code_executor.test.py` - Unit tests for code execution
- `src/rag/knowledge_base.py` - RAG system implementation and vector store management
- `src/rag/knowledge_base.test.py` - Unit tests for RAG functionality
- `src/parsers/document_parser.py` - Financial document parsing (10-K, statements)
- `src/parsers/document_parser.test.py` - Unit tests for document parsing
- `src/classifiers/question_classifier.py` - Question type classification logic
- `src/classifiers/question_classifier.test.py` - Unit tests for question classification
- `src/evaluation/benchmark_runner.py` - FinanceQA benchmark evaluation harness
- `src/evaluation/benchmark_runner.test.py` - Unit tests for evaluation system
- `src/utils/assumptions.py` - Assumption tracking and confidence scoring utilities
- `src/utils/assumptions.test.py` - Unit tests for assumption utilities
- `config/agent_config.py` - Configuration management and model selection
- `config/prompts.py` - Prompt templates for financial reasoning
- `requirements.txt` - Python dependencies specification
- `Dockerfile` - Container configuration for deployment
- `README.md` - Project documentation and setup instructions
- `docs/agent_card.md` - Agent capability and design documentation

### Notes

- Unit tests should be placed alongside the code files they test
- Use `pytest` for running tests: `pytest tests/` or `pytest path/to/specific/test.py`
- The project follows a modular architecture with clear separation of concerns
- All components should be designed with dependency injection for easy testing

## Tasks

- [ ] 1.0 Project Infrastructure Setup
  - [ ] 1.1 Create project directory structure (`src/`, `tests/`, `config/`, `docs/`, `data/`)
  - [ ] 1.2 Initialize Python project with `requirements.txt` including LangChain, OpenAI, ChromaDB, pandas, pytest
  - [ ] 1.3 Set up environment configuration files (`.env.example`, `.gitignore`)
  - [ ] 1.4 Create basic logging configuration and utility modules
  - [ ] 1.5 Set up pytest configuration and test directory structure

- [ ] 2.0 Core Agent Architecture Implementation
  - [ ] 2.1 Design and implement base `Agent` class with reasoning chain interface
  - [ ] 2.2 Create `ReasoningStep` data structure for step tracking and inspection
  - [ ] 2.3 Implement agent orchestration with tool selection and execution flow
  - [ ] 2.4 Add error handling and graceful degradation for tool failures
  - [ ] 2.5 Create agent state management for multi-step reasoning persistence
  - [ ] 2.6 Write comprehensive unit tests for core agent functionality

- [ ] 3.0 Model Management System
  - [ ] 3.1 Create abstract `ModelProvider` interface for LLM abstraction
  - [ ] 3.2 Implement `OllamaProvider` for local model integration
  - [ ] 3.3 Implement `OpenAIProvider` for API-based models
  - [ ] 3.4 Build model switching logic with configuration-based selection
  - [ ] 3.5 Add model response caching to reduce API calls
  - [ ] 3.6 Implement rate limiting and retry logic for API providers
  - [ ] 3.7 Write unit tests for all model provider implementations

- [ ] 4.0 Financial Calculation Tools
  - [ ] 4.1 Create base `FinancialTool` interface with calculation validation
  - [ ] 4.2 Implement NPV calculator with cash flow analysis and assumptions tracking
  - [ ] 4.3 Implement IRR calculator with iterative solving and edge case handling
  - [ ] 4.4 Build financial ratios calculator (P/E, debt-to-equity, current ratio, etc.)
  - [ ] 4.5 Create EBITDA adjustment calculator with standard accounting modifications
  - [ ] 4.6 Add input validation and meaningful error messages for all calculators
  - [ ] 4.7 Implement calculation step documentation and intermediate result tracking
  - [ ] 4.8 Write comprehensive unit tests covering edge cases and error conditions

- [ ] 5.0 RAG System Integration
  - [ ] 5.1 Set up ChromaDB vector store with financial knowledge embeddings
  - [ ] 5.2 Create document ingestion pipeline for financial knowledge corpus
  - [ ] 5.3 Implement semantic search with relevance scoring and filtering
  - [ ] 5.4 Build context ranking system based on question type classification
  - [ ] 5.5 Add citation tracking and source attribution for retrieved information
  - [ ] 5.6 Implement lazy loading and caching for vector store operations
  - [ ] 5.7 Create knowledge base management utilities (add/update/delete documents)
  - [ ] 5.8 Write unit tests for RAG pipeline and vector operations

- [ ] 6.0 Code Execution Environment
  - [ ] 6.1 Implement secure Python code execution using RestrictedPython
  - [ ] 6.2 Create code generation templates for common financial calculations
  - [ ] 6.3 Add timeout mechanisms and resource limits for code execution
  - [ ] 6.4 Build code validation and syntax checking before execution
  - [ ] 6.5 Implement result extraction and error handling from executed code
  - [ ] 6.6 Add code documentation generation with financial logic explanations
  - [ ] 6.7 Create execution environment cleanup and state isolation
  - [ ] 6.8 Write unit tests for code execution security and functionality

- [ ] 7.0 Document Processing Pipeline
  - [ ] 7.1 Implement PDF parsing for 10-K forms using pdfplumber
  - [ ] 7.2 Create section identification for financial statements (income, balance, cash flow)
  - [ ] 7.3 Build table extraction and data cleaning utilities
  - [ ] 7.4 Implement document structure analysis and content categorization
  - [ ] 7.5 Add data validation and consistency checks for extracted information
  - [ ] 7.6 Create document metadata extraction (company, period, filing type)
  - [ ] 7.7 Build caching system for parsed documents to avoid reprocessing
  - [ ] 7.8 Write unit tests with sample financial documents and edge cases

- [ ] 8.0 Question Classification System
  - [ ] 8.1 Implement question type classifier (tactical-basic, tactical-assumption, conceptual)
  - [ ] 8.2 Create keyword-based classification with financial domain terms
  - [ ] 8.3 Build confidence scoring system for classification decisions
  - [ ] 8.4 Implement missing information detection for assumption-based questions
  - [ ] 8.5 Add reasoning approach selection based on question classification
  - [ ] 8.6 Create assumption generation and confidence level assignment
  - [ ] 8.7 Build question complexity assessment for model selection
  - [ ] 8.8 Write unit tests with diverse question types from FinanceQA dataset

- [ ] 9.0 Evaluation and Benchmarking
  - [ ] 9.1 Download and prepare FinanceQA benchmark dataset
  - [ ] 9.2 Implement benchmark evaluation harness with exact match scoring
  - [ ] 9.3 Create performance metrics tracking (accuracy by question type)
  - [ ] 9.4 Build reasoning quality assessment tools for manual review
  - [ ] 9.5 Implement processing time measurement and performance profiling
  - [ ] 9.6 Add comparison utilities for baseline vs agent performance
  - [ ] 9.7 Create detailed evaluation reporting with failure analysis
  - [ ] 9.8 Write unit tests for evaluation metrics and scoring logic

- [ ] 10.0 Documentation and Containerization
  - [ ] 10.1 Write comprehensive README with setup instructions and usage examples
  - [ ] 10.2 Create agent card documenting capabilities, limitations, and design approach
  - [ ] 10.3 Document key trade-offs and architectural decisions
  - [ ] 10.4 Add API documentation for all public interfaces and tools
  - [ ] 10.5 Create Dockerfile with multi-stage build for production deployment
  - [ ] 10.6 Write docker-compose configuration for development environment
  - [ ] 10.7 Add example usage scripts and configuration templates
  - [ ] 10.8 Create troubleshooting guide and FAQ documentation