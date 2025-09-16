# PRD: LLM-Based Evaluation System for FinanceQA Benchmark

## Introduction/Overview

The current FinanceQA benchmark evaluation relies on exact string matching and basic normalization, which fails to recognize semantically correct answers that use different formatting, financial notation variations, or equivalent expressions. This leads to false negatives where correct answers are marked as incorrect due to format differences rather than semantic incorrectness.

This feature will replace the existing exact match evaluation with an LLM-based system that can judge semantic correctness while reusing existing infrastructure components. The goal is to improve evaluation accuracy for the FinanceQA benchmark with minimal implementation complexity.

## Goals

1. **Improve Evaluation Accuracy**: Reduce false negatives by recognizing semantically equivalent answers (e.g., "$1.5B" = "1500 million")
2. **Maintain Implementation Simplicity**: Reuse existing model_manager and evaluation infrastructure
3. **Enable Flexible Model Selection**: Allow users to choose evaluation model independently from agent model
4. **Provide Actionable Feedback**: Generate clear evaluation results that explain correctness decisions

## User Stories

**As a researcher evaluating my FinanceQA agent**, I want the benchmark to recognize when my agent gives correct answers in different formats, so that I get accurate performance metrics.

**As a developer running benchmark evaluations**, I want to configure which model performs the evaluation, so that I can balance cost and accuracy based on my needs.

**As a data scientist analyzing evaluation results**, I want to see why each answer was marked correct/incorrect, so that I can understand my agent's performance patterns.

## Functional Requirements

1. **LLM Scorer Implementation**: The system must provide an `LLMScorer` class that replaces `ExactMatchScorer` for semantic evaluation of predicted vs ground truth answers.

2. **Model Configuration**: The system must allow users to specify which model (via model_manager) performs evaluations through configuration options.

3. **Batch Processing**: The system must support batch evaluation of multiple answer pairs to optimize API usage and reduce latency.

4. **Binary Correctness Judgment**: The system must return a boolean correctness decision for each predicted/ground truth answer pair.

5. **Structured Evaluation Output**: The system must provide evaluation reasoning in a consistent format that explains the correctness decision.

6. **Financial Domain Awareness**: The system must understand financial notation equivalences (currency symbols, abbreviations, unit conversions).

7. **Error Handling**: The system must mark evaluation failures as errors and continue processing remaining examples rather than failing the entire evaluation.

8. **Existing Integration**: The system must integrate with existing `FinanceQAEvaluator` class and maintain compatibility with current benchmark runner workflow.

9. **Results Storage**: The system must extend existing benchmark output JSON to include LLM evaluation results alongside current metrics.

10. **Configuration Management**: The system must use existing configuration infrastructure to manage LLM evaluation settings.

## Non-Goals (Out of Scope)

- **Confidence scoring or uncertainty quantification** (focus on binary correctness only)
- **Partial credit or graduated scoring** (keep simple yes/no evaluation)
- **Multi-model consensus or validation** (single model evaluation)
- **Real-time cost tracking during evaluation** (post-evaluation cost reporting only)
- **Comparison with exact match results** (pure replacement, not hybrid approach)
- **Custom prompt engineering UI** (use fixed, well-tested prompts)
- **Caching or result persistence** (evaluate fresh each time for simplicity)

## Technical Considerations

**Component Reuse Strategy:**
- Leverage existing `model_manager` for LLM integration
- Extend `EvaluationResult` dataclass for new fields
- Reuse `FinanceQAEvaluator` as the main orchestrator
- Maintain existing `BenchmarkRunner` command-line interface

**Key Integration Points:**
- Replace `ExactMatchScorer` instantiation in `FinanceQAEvaluator.__init__()`
- Add LLM evaluation fields to `EvaluationResult` dataclass
- Extend benchmark output JSON structure
- Add model configuration options to existing config system

**Prompt Design:**
- Use structured JSON output format for consistent parsing
- Include few-shot examples for financial answer evaluation
- Design prompts to minimize hallucination and ensure reproducible results

## Success Metrics

**Primary Success Metrics:**
1. **Improved Recall**: Increase in correctly identified semantically equivalent answers compared to exact matching
2. **Maintained Precision**: No significant increase in false positives (incorrect answers marked correct)

**Secondary Success Metrics:**
3. **Implementation Simplicity**: Successful integration with <50 lines of changes to existing evaluation code
4. **Performance Adequacy**: Batch evaluation completes within reasonable time for development workflows

## Open Questions

1. **Prompt Template Selection**: Should we start with a simple instructional prompt or invest in few-shot examples from the start?

2. **Model Defaults**: What should be the default evaluation model when no specific model is configured?

3. **Error Rate Tolerance**: What percentage of LLM evaluation failures is acceptable before falling back to exact matching?

4. **JSON Schema Validation**: Should we implement strict JSON schema validation for LLM responses or use lenient parsing?

5. **Async vs Sync**: Should the initial implementation use async batch processing or start with synchronous evaluation for simplicity?

---

**Target Implementation**: This PRD targets a junior developer familiar with the existing FinanceQA codebase. The implementation should reuse existing patterns and infrastructure while adding minimal new complexity.