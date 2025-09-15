# Tasks: FinanceQA AI Agent

## Relevant Files

- `src/__init__.py` - Main package initialization with version and metadata
- `src/agent/__init__.py` - Agent module initialization
- `src/agent/core.py` - Main agent orchestration and reasoning chain implementation
- `src/models/__init__.py` - Models module initialization
- `src/models/model_manager.py` - Model switching infrastructure and abstract interfaces
- `src/models/ollama_provider.py` - Ollama local model provider implementation
- `docs/ollama_setup.md` - Step-by-step Ollama installation and configuration guide
- `src/tools/__init__.py` - Tools module initialization
- `src/tools/financial_calculator.py` - Simple financial calculation tools with ROI, ratios, NPV, IRR
- `src/tools/code_executor.py` - Secure Python code execution environment
- `src/rag/__init__.py` - RAG module initialization
- `src/rag/knowledge_base.py` - RAG system implementation and vector store management
- `src/parsers/__init__.py` - Parsers module initialization
- `src/parsers/document_parser.py` - Financial document parsing (10-K, statements)
- `src/classifiers/__init__.py` - Classifiers module initialization
- `src/classifiers/question_classifier.py` - Question type classification logic
- `src/evaluation/__init__.py` - Evaluation module initialization
- `src/evaluation/benchmark_runner.py` - FinanceQA benchmark evaluation harness
- `src/utils/__init__.py` - Utils module initialization
- `src/utils/assumptions.py` - Assumption tracking and confidence scoring utilities
- `tests/__init__.py` - Unit test location
- `config/agent_config.py` - Configuration management and model selection
- `config/prompts.py` - Prompt templates for financial reasoning
- `data/knowledge_base/` - Directory for financial knowledge corpus
- `data/datasets/` - Directory for FinanceQA and other datasets
- `data/sample_documents/` - Directory for sample financial documents
- `docs/` - Documentation directory
- `environment.yml` - Conda environment configuration
- `setup.py` - Python package setup configuration
- `pyproject.toml` - Modern Python project configuration with tool settings
- `Dockerfile` - Container configuration for deployment
- `README.md` - Project documentation and setup instructions
- `docs/agent_card.md` - Agent capability and design documentation
- `.gitignore` - Git ignore file for Python projects and data directories
- `.env.example` - Environment configuration template with all required variables
- `scripts/download_financeqa.py` - Script to download and prepare FinanceQA benchmark dataset from HuggingFace
- `src/evaluation/benchmark_runner.py` - FinanceQA benchmark evaluation harness with exact match scoring

### Notes

- The project follows a modular architecture with clear separation of concerns
- All components should be designed with dependency injection for easy testing
- Always use absolute path from the project root directory rather than relative path
- Avoid create mock for unit testing

## Tasks

### PHASE 1: CORE END-TO-END IMPLEMENTATION (MVP)

#### 1.0 Project Infrastructure Setup ✓ COMPLETED
- [x] 1.1 Create project directory structure (`src/`, `tests/`, `config/`, `docs/`, `data/`)
- [x] 1.2 Initialize Python project with `requirements.txt` including LangChain, OpenAI, ChromaDB, pandas, pytest
- [x] 1.3 Set up environment configuration files (`.env.example`, `.gitignore`)
- [x] 1.4 Create basic logging configuration and utility modules

#### 2.0 Dataset and Evaluation Foundation (CORE - 30 minutes)
- [x] 9.1 Download and prepare FinanceQA benchmark dataset
- [x] 9.2 Implement benchmark evaluation harness with exact match scoring
- [x] 9.3 Create performance metrics tracking (accuracy by question type)

#### 3.0 Basic Model Management (CORE - 45 minutes)
- [x] 3.1 Create abstract `ModelProvider` interface for LLM abstraction
- [x] 3.2 Implement `OllamaProvider` for local model integration
- [ ] 3.3 Implement `OpenAIProvider` for API-based models [SKIPPED]
- [x] 3.4 Build model switching logic with configuration-based selection

#### 4.0 Core Agent Architecture (CORE - 2 hours)
- [x] 2.1 Design and implement base `Agent` class with reasoning chain interface
- [x] 2.2 Create `ReasoningStep` data structure for step tracking and inspection
- [x] 8.1 Implement question type classifier (basic, assumption, conceptual)
- [x] 2.3 Implement basic agent orchestration with tool selection and execution flow
- [x] 2.4 Add error handling and graceful degradation for tool failures

#### 5.0 Essential Financial Tools (CORE - 1.5 hours)
- [x] 4.1 Create base `FinancialTool` interface with calculation validation
- [x] 4.4 Build financial ratios calculator (P/E, debt-to-equity, current ratio, etc.)
- [ ] 4.5 Create EBITDA adjustment calculator with standard accounting modifications [SKIPPED]
- [x] 4.6 Add input validation and meaningful error messages for all calculators

#### 6.0 Basic Document Processing (CORE - 1 hour) [SKIPPED]
- [ ] 7.1 Implement PDF parsing for 10-K forms using pdfplumber
- [ ] 7.2 Create section identification for financial statements (income, balance, cash flow)

#### 7.0 Question Classification System (CORE - 1 hour)
- [x] 8.2 Create keyword-based classification with financial domain terms
- [x] 8.4 Implement missing information detection for assumption-based questions
- [x] 8.5 Add reasoning approach selection based on question classification
- [x] 8.6 Create assumption generation and confidence level assignment

#### 8.0 Basic Documentation (CORE - 30 minutes)
- [x] 10.1 Write basic README with setup instructions and usage examples
- [ ] 10.2 Create agent card documenting capabilities, limitations, and design approach [SKIPPED]

**END OF PHASE 1 - SHOULD HAVE WORKING END-TO-END SYSTEM**

---

### PHASE 2: ACCURACY IMPROVEMENTS

#### 9.0 Enhanced Financial Calculations
- [x] 4.2 Implement NPV calculator with cash flow analysis and assumptions tracking
- [ ] 4.3 Implement IRR calculator with iterative solving and edge case handling
- [ ] 4.7 Implement calculation step documentation and intermediate result tracking

#### 10.0 Advanced Document Processing
- [ ] 7.3 Build table extraction and data cleaning utilities
- [ ] 7.4 Implement document structure analysis and content categorization
- [ ] 7.5 Add data validation and consistency checks for extracted information
- [ ] 7.6 Create document metadata extraction (company, period, filing type)

#### 11.0 Enhanced Question Classification
- [ ] 8.3 Build confidence scoring system for classification decisions
- [ ] 8.7 Build question complexity assessment for model selection

#### 12.0 Agent State Management
- [ ] 2.5 Create agent state management for multi-step reasoning persistence

---

### PHASE 3: ADVANCED FEATURES [OPTIONAL]

#### 13.0 RAG System Integration [OPTIONAL - HIGH EFFORT]
- [ ] [OPTIONAL] 5.1 Set up ChromaDB vector store with financial knowledge embeddings
- [ ] [OPTIONAL] 5.2 Create document ingestion pipeline for financial knowledge corpus
- [ ] [OPTIONAL] 5.3 Implement semantic search with relevance scoring and filtering
- [ ] [OPTIONAL] 5.4 Build context ranking system based on question type classification
- [ ] [OPTIONAL] 5.5 Add citation tracking and source attribution for retrieved information
- [ ] [OPTIONAL] 5.6 Implement lazy loading and caching for vector store operations
- [ ] [OPTIONAL] 5.7 Create knowledge base management utilities (add/update/delete documents)

#### 14.0 Code Execution Environment [OPTIONAL - HIGH EFFORT]
- [ ] [OPTIONAL] 6.1 Implement secure Python code execution using RestrictedPython
- [ ] [OPTIONAL] 6.2 Create code generation templates for common financial calculations
- [ ] [OPTIONAL] 6.3 Add timeout mechanisms and resource limits for code execution
- [ ] [OPTIONAL] 6.4 Build code validation and syntax checking before execution
- [ ] [OPTIONAL] 6.5 Implement result extraction and error handling from executed code
- [ ] [OPTIONAL] 6.6 Add code documentation generation with financial logic explanations
- [ ] [OPTIONAL] 6.7 Create execution environment cleanup and state isolation

#### 15.0 Advanced Model Management [OPTIONAL]
- [ ] [OPTIONAL] 3.5 Add model response caching to reduce API calls
- [ ] [OPTIONAL] 3.6 Implement rate limiting and retry logic for API providers

#### 16.0 Advanced Document Processing [OPTIONAL]
- [ ] [OPTIONAL] 7.7 Build caching system for parsed documents to avoid reprocessing

#### 17.0 Enhanced Evaluation [OPTIONAL]
- [ ] [OPTIONAL] 9.4 Build reasoning quality assessment tools for manual review
- [ ] [OPTIONAL] 9.5 Implement processing time measurement and performance profiling
- [ ] [OPTIONAL] 9.6 Add comparison utilities for baseline vs agent performance
- [ ] [OPTIONAL] 9.7 Create detailed evaluation reporting with failure analysis

---

### PHASE 4: PRODUCTION READINESS [OPTIONAL]

#### 18.0 Comprehensive Testing [OPTIONAL - HIGH EFFORT]
- [ ] [OPTIONAL] 2.6 Write comprehensive unit tests for core agent functionality
- [ ] [OPTIONAL] 3.7 Write unit tests for all model provider implementations
- [ ] [OPTIONAL] 4.8 Write comprehensive unit tests covering edge cases and error conditions
- [ ] [OPTIONAL] 5.8 Write unit tests for RAG pipeline and vector operations
- [ ] [OPTIONAL] 6.8 Write unit tests for code execution security and functionality
- [ ] [OPTIONAL] 7.8 Write unit tests with sample financial documents and edge cases
- [ ] [OPTIONAL] 8.8 Write unit tests with diverse question types from FinanceQA dataset
- [ ] [OPTIONAL] 9.8 Write unit tests for evaluation metrics and scoring logic

#### 19.0 Deployment and Containerization [OPTIONAL]
- [ ] [OPTIONAL] 10.5 Create Dockerfile with multi-stage build for production deployment
- [ ] [OPTIONAL] 10.6 Write docker-compose configuration for development environment
- [ ] [OPTIONAL] 10.7 Add example usage scripts and configuration templates

#### 20.0 Advanced Documentation [OPTIONAL]
- [ ] [OPTIONAL] 10.3 Document key trade-offs and architectural decisions
- [ ] [OPTIONAL] 10.4 Add API documentation for all public interfaces and tools
- [ ] [OPTIONAL] 10.8 Create troubleshooting guide and FAQ documentation
