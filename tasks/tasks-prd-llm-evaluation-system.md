# Tasks: LLM-Based Evaluation System for FinanceQA Benchmark

## Relevant Files

- `src/evaluation/llm_scorer.py` - New LLM-based scorer class that replaces ExactMatchScorer functionality
- `src/evaluation/benchmark_runner.py` - Main evaluation system - needs updates to integrate LLMScorer
- `config/model_config.json` - Model configuration - needs evaluation model settings
- `src/models/model_manager.py` - Model management - may need minor updates for evaluation context

### Notes

- The implementation focuses on minimal changes to existing code while replacing exact matching with LLM evaluation
- Reuses existing model_manager infrastructure and configuration patterns
- Maintains backward compatibility with current BenchmarkRunner CLI interface

## Tasks

- [ ] 1.0 Create LLM Scorer Implementation
  - [ ] 1.1 Create LLMScorer class with same interface as ExactMatchScorer
  - [ ] 1.2 Design and implement financial evaluation prompt template with JSON output format
  - [ ] 1.3 Implement prompt formatting and few-shot examples for financial domain
  - [ ] 1.4 Add JSON response parsing with error handling for malformed responses
  - [ ] 1.5 Implement compute_llm_match method that returns boolean correctness decision

- [ ] 2.0 Integrate LLM Scorer with Evaluation System
  - [ ] 2.1 Update FinanceQAEvaluator.__init__() to accept LLMScorer instead of ExactMatchScorer
  - [ ] 2.2 Modify FinanceQAEvaluator.evaluate_agent() to use LLM evaluation instead of exact/normalized matching
  - [ ] 2.3 Update evaluation loop to handle async LLM calls if needed
  - [ ] 2.4 Ensure BenchmarkRunner continues to work with new evaluation system

- [ ] 3.0 Update Data Structures for LLM Evaluation Results
  - [ ] 3.1 Extend EvaluationResult dataclass to include LLM evaluation fields (llm_match, llm_reasoning)
  - [ ] 3.2 Update FinanceQAEvaluator.compute_metrics() to calculate LLM-based accuracy metrics
  - [ ] 3.3 Modify benchmark output JSON structure to include LLM evaluation results
  - [ ] 3.4 Update BenchmarkRunner result saving and summary printing to show LLM metrics

- [ ] 4.0 Add Configuration Support for Evaluation Models
  - [ ] 4.1 Extend model_config.json to include evaluation model configuration section
  - [ ] 4.2 Add evaluation model selection parameter to FinanceQAEvaluator constructor
  - [ ] 4.3 Update BenchmarkRunner to pass evaluation model configuration to evaluator
  - [ ] 4.4 Implement fallback to default model when evaluation model not specified

- [ ] 5.0 Implement Batch Processing and Error Handling
  - [ ] 5.1 Add batch evaluation method to LLMScorer for processing multiple answer pairs efficiently
  - [ ] 5.2 Implement error handling in LLMScorer that marks failures and continues evaluation
  - [ ] 5.3 Add retry logic for transient LLM API failures with exponential backoff
  - [ ] 5.4 Update evaluation pipeline to collect and report LLM evaluation error rates
  - [ ] 5.5 Test end-to-end evaluation with both successful and failed LLM evaluations