## Relevant Files

- `src/evaluation/shared_evaluation_utils.py` - ✅ CREATED - Common utilities for LLM-based evaluation including data classes, CLI parsing, model setup, and LLM scorer utilities
- `src/evaluation/run_agent_evaluation.py` - ✅ CREATED - Standalone script for running agents and generating result files (no evaluation)
- `src/evaluation/run_llm_evaluation.py` - ✅ CREATED - Standalone script for LLM-based evaluation of existing result files
- `src/evaluation/benchmark_runner.py` - ✅ CREATED - Refactored orchestrator that coordinates agent and LLM evaluation workflows
- `src/evaluation/benchmark_runner_original.py` - ✅ CREATED - Backup of original benchmark_runner.py
- `src/evaluation/llm_scorer.py` - Primary scorer for all evaluation (replaces exact match scoring)

### Notes

- The refactoring moves to LLM-based evaluation only, deprecating exact match scoring
- All scripts maintain CLI interfaces compatible with existing usage patterns
- Result file format updated to remove deprecated exact_match and normalized_match fields
- Focus on functional implementation without unit test requirements
- DummyAgent removed as it's only used for testing the evaluation harness

## Tasks

- [x] 1.0 Extract Common Utilities and Data Classes
  - [x] 1.1 Create `shared_evaluation_utils.py` with data classes (EvaluationExample, AgentResponse, EvaluationResult)
  - [x] 1.2 Extract FinancialAgent protocol to shared utilities (DummyAgent deprecated)
  - [x] 1.3 Extract common CLI argument parsing logic for reuse across scripts
  - [x] 1.4 Extract result file serialization/deserialization functions
  - [x] 1.5 Extract dataset loading functionality from FinanceQAEvaluator
  - [x] 1.6 Remove ExactMatchScorer dependencies and use only LLM-based evaluation
- [x] 2.0 Create Standalone Agent Evaluation Script
  - [x] 2.1 Create `run_agent_evaluation.py` with main() function and CLI argument parser
  - [x] 2.2 Implement agent creation logic (import from src.agent.core, model manager setup)
  - [x] 2.3 Implement agent execution using shared utilities (load dataset, run agent)
  - [x] 2.4 Implement result file generation with JSON format (remove exact_match and normalized_match fields)
  - [x] 2.5 Add support for all evaluation types: full, quick, by-type
  - [x] 2.6 Add error handling for agent failures and include in result files
  - [x] 2.7 Add progress bars and timing information
  - [x] 2.8 Add executable shebang and make script directly runnable
- [x] 3.0 Create Standalone LLM Evaluation Script
  - [x] 3.1 Create `run_llm_evaluation.py` with main() function and CLI argument parser
  - [x] 3.2 Implement result file loading and validation (check format, handle missing fields)
  - [x] 3.3 Implement LLM scorer integration using existing llm_scorer.py
  - [x] 3.4 Add configuration options for handling failed agent examples (skip, process, flag)
  - [x] 3.5 Implement result file updating (add llm_match and llm_reasoning fields)
  - [x] 3.6 Add model configuration support for evaluation-specific models
  - [x] 3.7 Add summary statistics and reporting for LLM evaluation results
  - [x] 3.8 Add progress tracking for LLM evaluation process
- [x] 4.0 Refactor Benchmark Runner as Orchestrator
  - [x] 4.1 Backup current benchmark_runner.py as benchmark_runner_original.py
  - [x] 4.2 Create new benchmark_runner.py that imports run_agent_evaluation and run_llm_evaluation
  - [x] 4.3 Implement orchestration logic to call agent evaluation first, then LLM evaluation
  - [x] 4.4 Maintain exact same CLI interface as original benchmark_runner
  - [x] 4.5 Add logic to conditionally run LLM evaluation based on --use-llm-evaluation flag
  - [x] 4.6 Implement temporary file handling between agent and LLM evaluation steps
  - [x] 4.7 Add error handling for orchestration failures (agent fails, LLM fails, etc.)
  - [x] 4.8 Preserve all existing functionality (progress bars, metrics, output formatting)
- [ ] 5.0 Ensure Functional Compatibility
  - [ ] 5.1 Test agent evaluation script produces agent results (without exact match scoring)
  - [ ] 5.2 Test LLM evaluation script correctly processes existing result files and adds LLM scoring
  - [ ] 5.3 Test orchestrated workflow produces LLM-based evaluation results
  - [ ] 5.4 Verify all CLI arguments work correctly across all three scripts
  - [ ] 5.5 Test error handling scenarios (missing files, malformed results, agent failures)
  - [ ] 5.6 Verify result file format compatibility with existing analysis tools (with deprecated fields removed)
- [ ] 6.0 Verify Integration and Performance
  - [ ] 6.1 Run full LLM-based evaluation using refactored implementation
  - [ ] 6.2 Test separate execution workflow (agent → manual analysis → LLM evaluation)
  - [ ] 6.3 Verify performance matches or exceeds original benchmark_runner speed
  - [ ] 6.4 Test with different model configurations and evaluation settings
  - [ ] 6.5 Validate that existing result files can be processed by new LLM evaluation script
  - [ ] 6.6 Document usage examples for all three execution patterns