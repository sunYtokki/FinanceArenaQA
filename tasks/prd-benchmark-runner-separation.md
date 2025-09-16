# PRD: Benchmark Runner Separation

## Introduction/Overview

This feature refactors the existing `benchmark_runner.py` into two separate Python scripts to decouple agent evaluation from LLM-based scoring. Currently, the benchmark runner combines agent execution and evaluation scoring into a single process, which limits flexibility for batch processing multiple result files with different LLM evaluation models, debugging agent results before LLM evaluation, and running evaluations on different machines/environments.

The goal is to create two independent scripts: one for running agents to generate result files, and another for performing LLM evaluation on existing result files.

## Goals

1. **Decouple Processes**: Separate agent evaluation from LLM scoring for improved workflow flexibility
2. **Enable Batch Processing**: Allow LLM evaluation to process multiple result files efficiently
3. **Improve Debugging**: Enable analysis of agent results before LLM evaluation
4. **Resource Optimization**: Allow running agent and LLM evaluation on different machines/environments
5. **Maintain Compatibility**: Deprecate the current combined approach while keeping it temporarily available

## User Stories

**As a researcher**, I want to run agent evaluation once and then experiment with different LLM evaluation models, so that I can compare evaluation approaches without re-running expensive agent inference.

**As a developer**, I want to debug and analyze agent results before running LLM evaluation, so that I can identify issues in agent responses and fix them before scoring.

**As a data scientist**, I want to batch process multiple result files with LLM evaluation, so that I can efficiently evaluate large numbers of experiments.

**As a team member**, I want to run agent evaluation on a GPU machine and LLM evaluation on a different machine, so that I can optimize resource usage across our infrastructure.

## Functional Requirements

### Agent Evaluation Script (`run_agent_evaluation.py`)

1. **FR-1**: The script must accept the same command-line arguments as the current benchmark runner for agent evaluation (dataset-path, output-dir, split, evaluation-type, num-samples, question-type, output-file)

2. **FR-2**: The script must run agent evaluation and generate result files in the exact same JSON format as currently produced by `benchmark_runner.py`

3. **FR-3**: The script must support all current evaluation types: full, quick, and by-type

4. **FR-4**: The script must handle agent errors gracefully and include error information in the result file

5. **FR-5**: The script must include all current metadata in the result file: timestamps, processing times, agent responses, confidence scores, and reasoning steps

6. **FR-6**: The script must support both sync and async agent implementations as the current system does

### LLM Evaluation Script (`run_llm_evaluation.py`)

7. **FR-7**: The script must accept a single result file path as input for LLM evaluation

8. **FR-8**: The script must read existing result files in the current JSON format without modification

9. **FR-9**: The script must extract predicted answers and ground truth from result files for LLM scoring

10. **FR-10**: The script must use only the `LLMScorer` class from `llm_scorer.py` for evaluation (ExactMatchScorer is deprecated)

11. **FR-11**: The script must update result files with LLM evaluation scores (`llm_match` and `llm_reasoning` fields)

12. **FR-12**: The script must handle cases where agent evaluation failed for some examples by providing configurable options:
    - Skip failed examples in LLM evaluation
    - Run LLM evaluation only on successful examples
    - Allow LLM evaluation to handle/flag failed cases

13. **FR-13**: The script must accept model configuration for LLM evaluation (evaluation-model parameter)

14. **FR-14**: The script must generate updated result files with LLM scores while preserving all original data

15. **FR-15**: The script must provide summary statistics showing LLM evaluation results

### Orchestrated Workflow

16. **FR-16**: The new `benchmark_runner.py` must act as an orchestrator that internally calls both `run_agent_evaluation.py` and `run_llm_evaluation.py`

17. **FR-17**: The orchestrated workflow must maintain the exact same command-line interface as the original `benchmark_runner.py`

18. **FR-18**: The orchestrated workflow must provide seamless combined execution while enabling the option for separate execution

## Non-Goals (Out of Scope)

1. **NG-1**: Modifying the result file format - must maintain exact compatibility with current JSON structure
2. **NG-2**: Supporting multiple result files as input to LLM evaluation in initial version
3. **NG-3**: Creating a new unified CLI command - will use separate Python scripts
4. **NG-4**: Implementing real-time streaming evaluation between agent and LLM components
5. **NG-5**: Adding new evaluation metrics beyond what currently exists

## Deprecated Components

1. **ExactMatchScorer**: Will be deprecated in favor of LLM-based evaluation only
2. **DummyAgent**: Will be removed as it's only used for testing the evaluation harness
3. **Exact Match and Normalized Match**: These scoring methods will be replaced by LLM evaluation

## Design Considerations

### File Structure
```
src/evaluation/
├── benchmark_runner.py          # New orchestrator (combines both workflows)
├── run_agent_evaluation.py      # Standalone agent evaluation script
├── run_llm_evaluation.py        # Standalone LLM evaluation script
├── llm_scorer.py               # Existing LLM scorer (unchanged)
└── shared_evaluation_utils.py   # Common utilities shared by all components
```

### Result File Compatibility
- Must maintain exact JSON schema compatibility with existing result files
- LLM evaluation script should add fields to existing structure, not replace it
- Original agent evaluation metadata must be preserved

### Error Handling Strategy
- Agent evaluation script: Continue current error handling, save errors in result file
- LLM evaluation script: Provide command-line options for handling failed examples
- Both scripts: Use consistent exit codes and error messaging

## Technical Considerations

1. **Code Reuse**: Extract common functionality from `benchmark_runner.py` into `shared_evaluation_utils.py` to avoid duplication

2. **Dependencies**: Scripts should use LLMScorer and model manager dependencies, removing ExactMatchScorer dependencies

3. **Configuration**: Leverage existing configuration loading patterns from `benchmark_runner.py`

4. **Import Structure**: Maintain compatibility with existing imports for agent creation and model management

5. **Deprecation Strategy**: Remove ExactMatchScorer and DummyAgent from shared utilities, use only LLM-based evaluation

## Success Metrics

1. **Functional Equivalence**: New separated scripts produce identical results to current combined approach (100% compatibility)

2. **Performance**: Agent evaluation script performance matches or exceeds current benchmark runner speed

3. **Flexibility**: Successfully run LLM evaluation on result files generated by different agent configurations

4. **Usage Adoption**: Development team successfully migrates to separated workflow within 2 weeks

5. **Error Reduction**: Reduced debugging time for evaluation issues by 50% due to separation of concerns

## Open Questions

1. **Transition Timeline**: How long should the deprecation period be for the original `benchmark_runner.py`?

2. **Batch Processing Priority**: Should batch processing of multiple result files be included in the initial implementation or saved for a future enhancement?

3. **Configuration Management**: Should the LLM evaluation script use the same model configuration file as agent evaluation, or support a separate evaluation-specific config?

4. **Output Naming**: Should LLM-evaluated result files have a different naming convention to distinguish them from agent-only results?

5. **Validation**: Should the LLM evaluation script validate that input result files were generated by compatible agent evaluation versions?