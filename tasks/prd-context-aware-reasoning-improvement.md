# PRD: Context-Aware Reasoning Chain Improvement

## Introduction/Overview

The current FinanceQA agent reasoning chain is showing degraded performance. Without reasoning chain 0.42 on basic questions, 0.54 for conceptual whereas with reasoning chain 0.34 on basic questiona 0.48 on conceptual. Particularly on assumption-type questions (2.17% accuracy vs 34% for basic questions). The 15-step complex reasoning chain defaults to common financial adjustments (like stock-based compensation) instead of leveraging contextual hints in financial documents (like detailed lease expense breakouts signaling lease-based adjustments).

This feature will implement a simplified, context-aware reasoning system that uses RAG-based retrieval to identify company and context-specific financial adjustment patterns, improving both accuracy and execution performance.

## Goals

2. **Primary Goal**: Reduce reasoning chain execution time by 40%+ through streamlined 2~5-step process (down from 15 steps)
1. **Seconday Goal**: Increase assumption question accuracy from 2.17% to 25%+ by implementing context-aware financial pattern detection
3. **Tertiary Goal**: Maintain or improve overall accuracy across all question types while simplifying the system

## User Stories

1. **As a financial analyst**, I want the agent to recognize when lease expenses are broken out in detail so that it applies lease-based EBITDA adjustments instead of defaulting to stock-based compensation adjustments.

2. **As a researcher evaluating the system**, I want consistent, context-appropriate financial calculations so that benchmark results reflect real-world financial analysis accuracy.

3. **As a developer maintaining the system**, I want a simple, linear reasoning chain so that debugging and improvements are straightforward and predictable.

4. **As an end user**, I want faster response times with more accurate financial calculations, especially for questions requiring financial statement adjustments.

## Functional Requirements

### Core Reasoning Chain Requirements

1. **FR1**: The system must implement a less than 5-step linear reasoning chain: Question Analysis → Context RAG → Calculate → Validate → Answer
2. **FR2**: The system must eliminate complex branching logic and reduce total reasoning steps from 15 to 5
3. **FR3**: The system must remove placeholder steps (sensitivity analysis, unused classification branches)
4. **FR4**: The system must maintain backward compatibility with existing tool interfaces and model managers

### Context-Aware RAG Requirements

5. **FR5**: The system must implement a ChromaDB vector store containing financial adjustment patterns with structured evidence and metadata
6. **FR6**: The system must detect context signals in financial documents including table data (lease expenses, stock-based compensation, etc.)
7. **FR7**: The system must retrieve relevant adjustment patterns and return structured evidence with provenance
8. **FR8**: The system must use confidence thresholds to decide between RAG-based adjustments and assumption fallbacks

### Financial Pattern Detection Requirements

9. **FR9**: The system must return structured adjustment specifications with evidence provenance and confidence scores
10. **FR10**: The system must support table-aware chunking for financial data ingestion with proper metadata
11. **FR11**: The system must use simple hybrid retrieval (keyword filtering + vector search) without complex reranking
12. **FR12**: The system must provide basic reasoning transparency with evidence citations while maintaining assumption fallbacks

### Performance Requirements

13. **FR13**: The system must execute reasoning chains in under 60% of current execution time
14. **FR14**: The system must maintain sub-200ms response time for RAG context retrieval
15. **FR15**: The system must handle concurrent reasoning requests without performance degradation

## Non-Goals (Out of Scope)

1. **Advanced NLP**: Complex natural language understanding beyond simple pattern matching
2. **Sophisticated ML**: Deep learning models for context detection (keeping it simple with ChromaDB)
3. **Dynamic Learning**: Online learning or automatic pattern discovery from new documents
4. **UI Changes**: Any modifications to the agent's user interface or interaction patterns
5. **Model Training**: Fine-tuning or retraining of underlying LLMs
6. **Multi-Modal**: Processing of charts, graphs, or non-text financial data

## Technical Considerations

### Integration Requirements
- Must integrate with existing `src/agent/core.py` reasoning chain infrastructure
- Must work with current Ollama/OpenAI model manager without modifications
- Must utilize existing ChromaDB setup from RAG system (`src/rag/`)

### Data Requirements
- Structured financial evidence storage: RAGEvidence with text, page, section, chunk_type, confidence
- Adjustment specifications: type, scope, basis, source_ids, confidence
- Table-aware ingestion with markdown format and metadata (units, period, section)
- Context-only indexing with no ground truth contamination

### Performance Constraints
- ChromaDB vector operations must complete within 200ms
- Total reasoning chain must execute 40% faster than current 15-step process
- Memory usage must not exceed current system requirements

## Success Metrics

### Primary Success Metrics (Priority 1)
1. **Execution Time Reduction**: Reasoning chain execution time decreased by 40%+
2. **Response Latency**: End-to-end response time under 3 seconds for complex questions

### Secondary Success Metrics (Priority 2)
1. **Assumption Accuracy**: Assumption question accuracy improved from 2.17% to 25%+
2. **Context Detection Rate**: 80%+ accuracy in detecting relevant contextual signals

### Tertiary Success Metrics (Priority 3)
1. **Overall Accuracy Maintenance**: Basic question accuracy maintained at 34%+
2. **Conceptual Question Stability**: Conceptual question accuracy maintained at 48%+

### Monitoring Metrics
1. **RAG Retrieval Success Rate**: 95%+ successful context pattern retrieval
2. **System Reliability**: <1% reasoning chain failures
3. **Pattern Coverage**: 90%+ of assumption questions matched to relevant patterns

## Implementation Phases

### Phase 1: Reasoning Chain Simplification (Week 1)
- Reduce reasoning steps from 15 to 5
- Remove complex branching and placeholder steps
- Implement linear execution flow
- Maintain existing functionality

### Phase 2: ChromaDB Context Store (Week 2)
- Set up ChromaDB collection for financial patterns
- Populate with company-specific adjustment patterns
- Implement context signal detection
- Create retrieval interface

### Phase 3: Context-Aware Integration (Week 3)
- Integrate RAG retrieval into reasoning chain
- Replace assumption generation with context-based selection
- Implement pattern ranking and selection logic
- Add reasoning transparency

### Phase 4: Validation & Optimization (Week 4)
- Performance testing and optimization
- Accuracy validation on assumption questions
- Edge case handling and error recovery
- Documentation and deployment preparation

## Open Questions

1. **Pattern Dataset**: Use context fields from test data but ensure no ground truth contamination
  - Index only context text and tables, never answers or ground truth labels
  - Implement guard to reject JSON keys like 'answer', 'ground_truth', 'cot'
2. **Structured Output**: RAG must return typed evidence and adjustment specifications
  - RAGResult with signals[], evidence[], proposed_adjustments[]
  - AdjustmentSpec with type, scope, basis, source_ids, confidence
3. **Fallback Strategy**: Layer assumptions rather than replace them
  - Use confidence thresholds to decide RAG vs assumption path
  - Maintain single execution path (no A/B branching)
4. **Hybrid Retrieval**: Simple keyword filtering + vector search
  - No complex reranking or cross-encoders
  - Basic confidence scoring with section bonuses
5. **Table Support**: Ingest tables as markdown with metadata
  - Store chunk_type, section, page, units, period
  - Basic table parsing without complex understanding

## Dependencies

- Existing ChromaDB infrastructure (`src/rag/`)
- Current agent core reasoning framework (`src/agent/core.py`)
- Financial calculation tools (`src/tools/`)
- Evaluation framework for accuracy testing

## Risk Mitigation

1. **Accuracy Regression**: Implement A/B testing between old and new reasoning chains
2. **Performance Issues**: Set strict timeout limits and fallback to simplified reasoning
3. **Context Pattern Gaps**: Maintain fallback to current assumption generation logic
4. **Integration Complexity**: Implement feature flags for gradual rollout