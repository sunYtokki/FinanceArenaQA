# Tasks for Context-Aware Reasoning Chain Improvement

Based on the PRD analysis and current codebase assessment, I have generated the high-level tasks required to implement this feature.

## Current Codebase Analysis

**Existing Infrastructure:**
- Complex 15-step reasoning chain in `src/agent/core.py` with branching logic
- Basic RAG infrastructure in `src/rag/` (minimal implementation)
- Financial tools in `src/tools/financial_calculator.py`
- Test data in `data/datasets/financeqa/test.jsonl` with context fields
- Evaluation results showing poor assumption question performance (2.17% accuracy)

**Key Issues Identified:**
- Reasoning chain has 15 steps with complex branching and placeholder implementations
- No context-aware pattern detection for financial adjustments
- Default to common adjustments (SBC) instead of contextual signals (lease expenses)
- No RAG integration for company-specific financial patterns

## High-Level Tasks

- [ ] 1.0 Streamline Core Reasoning Chain Architecture
  - [ ] 1.1 Modify existing `src/agent/core.py` to implement 5-step linear flow
  - [ ] 1.2 Implement typed interfaces between steps for structured data flow
  - [ ] 1.3 Remove complex branching logic, placeholder steps, and unused classification
  - [ ] 1.4 Maintain backward compatibility with existing tool interfaces and model managers
  - [ ] 1.5 Add execution time tracking and basic performance metrics
  - [ ] 1.6 Create structured step outputs: RAGResult, AdjustmentSpec data flow

- [ ] 2.0 Implement Context-Aware RAG System
  - [ ] 2.1 Create structured data types in `src/rag/data_types.py` (RAGEvidence, AdjustmentSpec, RAGResult)
  - [ ] 2.2 Set up ChromaDB collection with table-aware chunking and metadata storage
  - [ ] 2.3 Implement evidence storage with chunk_type, section, page, units, period fields
  - [ ] 2.4 Create simple hybrid retrieval: keyword filtering + vector search
  - [ ] 2.5 Add basic confidence scoring with section bonuses (no complex reranking)
  - [ ] 2.6 Implement simple caching for reasonable response times

- [ ] 3.0 Build Financial Pattern Detection Engine
  - [ ] 3.1 Create structured adjustment detector returning AdjustmentSpec with confidence
  - [ ] 3.2 Implement table-aware data extraction with markdown parsing and metadata
  - [ ] 3.3 Build context-only loader with ground truth contamination guards
  - [ ] 3.4 Add confidence threshold logic for RAG vs assumption fallback decisions
  - [ ] 3.5 Implement keyword mapping for adjustment types (leases, SBC, impairments)
  - [ ] 3.6 Create simple evidence scoring without complex verification gates

- [ ] 4.0 Integrate Context-Aware Logic into Reasoning Chain
  - [ ] 4.1 Integrate structured RAG into step 2 returning RAGResult objects
  - [ ] 4.2 Layer RAG adjustments over assumptions (don't replace completely)
  - [ ] 4.3 Implement single-path execution with confidence-based RAG vs fallback
  - [ ] 4.4 Add basic reasoning transparency with evidence citations
  - [ ] 4.5 Create fallback to assumption generation when confidence below threshold
  - [ ] 4.6 Configure simple thresholds and keyword mappings in YAML
  - [ ] 4.7 Ensure calculators can consume AdjustmentSpec objects

- [ ] 5.0 Performance Testing and Validation
  - [ ] 5.1 Create evaluation framework measuring hit rates and adoption metrics
  - [ ] 5.2 Compare old vs new reasoning chains (single path, no A/B branching)
  - [ ] 5.3 Validate assumption question accuracy improvement (target: 2.17% → 25%+)
  - [ ] 5.4 Measure execution time reduction (target: 40%+ improvement)
  - [ ] 5.5 Test evidence retrieval success rate and confidence calibration
  - [ ] 5.6 Validate overall accuracy maintenance across question types
  - [ ] 5.7 Monitor RAG adoption rate vs fallback usage
  - [ ] 5.8 Test table parsing and metadata extraction accuracy
  - [ ] 5.7 Create migration script `scripts/migrate_reasoning_chain.py` for deployment
  - [ ] 5.8 Add monitoring and logging for pattern effectiveness tracking

## Relevant Files

- `src/agent/core.py` - Modified reasoning chain with 5-step linear flow
- `src/agent/core.test.py` - Unit tests for simplified reasoning chain
- `src/rag/context_patterns.py` - ChromaDB-based structured evidence storage
- `src/rag/context_patterns.test.py` - Unit tests for evidence retrieval
- `src/rag/financial_context_detector.py` - Structured adjustment detection with confidence scoring
- `src/rag/financial_context_detector.test.py` - Unit tests for context detection
- `src/rag/data_types.py` - RAGEvidence, AdjustmentSpec, RAGResult dataclasses
- `src/data/pattern_loader.py` - Load context data with ground truth guards
- `src/data/pattern_loader.test.py` - Unit tests for pattern loading
- `src/agent/context_aware_agent.py` - Main agent with structured RAG integration
- `src/agent/context_aware_agent.test.py` - Integration tests for context-aware agent
- `src/evaluation/context_evaluation.py` - Accuracy and hit rate testing
- `src/evaluation/context_evaluation.test.py` - Unit tests for evaluation metrics
- `config/context_patterns.yaml` - Confidence thresholds and keyword mappings

### Notes

- Unit tests should be placed alongside the code files they test
- Use `pytest` to run tests
- ChromaDB collections will be stored in existing `src/rag/` infrastructure
- Pattern data will be extracted from `data/datasets/financeqa/test.jsonl` context fields

## Tasks