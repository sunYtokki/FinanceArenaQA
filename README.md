# FinanceQA AI Agent

A multi-step reasoning AI agent designed to improve upon the 54.1% baseline performance on the FinanceQA benchmark through intelligent question classification, financial calculation tools, and adaptive reasoning strategies.

## Overview

This agent tackles complex financial analysis tasks that traditional LLMs struggle with by:
- **Question Classification**: Automatically routes questions to appropriate reasoning strategies
- **Multi-step Reasoning**: Decomposes complex problems into manageable steps
- **Financial Tools**: Specialized calculators for NPV, IRR, ratios, and ROI
- **Error Handling**: Graceful degradation with LLM fallbacks
- **Reasoning Chain Inspection**: Complete transparency into decision-making process

## Quick Start

### Prerequisites

- Python 3.9+
- Conda (recommended) or pip
- Ollama (for local models) or OpenAI API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Veris
   ```

2. **Create and activate environment**:
   ```bash
   conda env create -f environment.yml
   conda activate financeqa-agent
   ```

3. **Set up Ollama** (recommended for development):
   ```bash
   # Follow the detailed guide
   cat docs/ollama_setup.md

   # Quick setup:
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve
   ollama pull llama2
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your model configurations
   ```

### Basic Usage

```python
import asyncio
from src.agent.core import Agent
from src.models.model_manager import ModelManager, create_model_manager_from_config
from src.tools.financial_calculator import FinancialCalculator

# Configure model manager
config = {
    "providers": {
        "ollama": {
            "type": "ollama",
            "model_name": "llama2",
            "base_url": "http://localhost:11434",
            "temperature": 0.1
        }
    },
    "default_provider": "ollama"
}

async def main():
    # Initialize components
    model_manager = create_model_manager_from_config(config)
    financial_calc = FinancialCalculator()

    # Create agent
    agent = Agent(model_manager, tools=[financial_calc])

    # Ask a financial question
    question = "Calculate the ROI if I invest $1000 and gain $1200"

    # Get reasoning chain
    chain = await agent.answer_question(question)

    # Print results
    print(f"Question: {question}")
    print(f"Answer: {chain.final_answer}")
    print(f"Confidence: {chain.confidence_score}")
    print(f"Steps: {len(chain.steps)}")

# Run the example
asyncio.run(main())
```

## End-to-End Testing

### Unit Tests

Run individual component tests:

```bash
# Test financial calculator
pytest src/tools/financial_calculator.test.py -v

# Test model management
pytest src/models/ollama_provider.test.py -v

# Test question classifier
pytest src/classifiers/question_classifier.test.py -v

# Run all unit tests
pytest -v
```

### Integration Tests

Test complete system integration:

```bash
# Requires running Ollama server
pytest src/models/ollama_provider.test.py::TestOllamaProviderIntegration -v -m integration

# End-to-end agent testing
python scripts/test_e2e.py
```

### FinanceQA Benchmark Evaluation

Evaluate agent performance on the FinanceQA benchmark:

```bash
# Download FinanceQA dataset (if not already done)
python scripts/download_financeqa.py

# Full dataset evaluation (all examples)
python src/evaluation/benchmark_runner.py \
    --evaluation-type full \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Quick evaluation (50 samples for testing)
python src/evaluation/benchmark_runner.py \
    --evaluation-type quick \
    --num-samples 50 \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Evaluate specific question types
python src/evaluation/benchmark_runner.py \
    --evaluation-type by-type \
    --question-type "calculation" \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# View results
ls results/
cat results/financeqa_full_evaluation_test_*.json
```

## Model Performance Evaluation

### Performance Metrics

The agent tracks multiple performance indicators:

1. **Accuracy Metrics**:
   - Exact match accuracy on FinanceQA benchmark
   - Question type classification accuracy
   - Tool selection success rate

2. **Quality Metrics**:
   - Reasoning chain completeness
   - Step success/failure rates
   - Confidence calibration

3. **Efficiency Metrics**:
   - Average response time
   - Token usage per question
   - Tool execution time

### Running Performance Tests

```python
from src.evaluation.benchmark_runner import BenchmarkRunner

# Initialize benchmark runner with custom dataset path
runner = BenchmarkRunner(
    dataset_path="data/datasets/financeqa",
    output_dir="results/"
)

# Run full evaluation on test split (all examples)
results = runner.run_full_evaluation(
    agent=your_agent,
    split="test",
    verbose=True
)

# Run quick evaluation for development testing
results = runner.run_quick_evaluation(
    agent=your_agent,
    split="test",
    num_samples=100,
    verbose=True
)

# Evaluate specific question types
results = runner.run_by_question_type(
    agent=your_agent,
    question_type="calculation",
    split="test"
)

# Print performance summary
metrics = results['metrics']
print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.1%}")
print(f"Normalized Match Accuracy: {metrics['normalized_match_accuracy']:.1%}")
print(f"Error Rate: {metrics['error_rate']:.1%}")
print(f"Total Examples: {metrics['total_examples']}")
```

### Performance Baseline

Target performance improvements over baseline:

| Metric | Baseline | Target | Current Status |
|--------|----------|---------|---------------|
| FinanceQA Accuracy | 54.1% | >65% | *In Progress* |
| Question Classification | N/A | >85% | *Implemented* |
| Tool Integration | N/A | >90% | *Implemented* |
| Error Recovery | N/A | >95% | *Implemented* |

## Architecture

### Core Components

```
src/
├── agent/
│   └── core.py                 # Main Agent class with reasoning chains
├── models/
│   ├── model_manager.py        # Model abstraction and switching
│   └── ollama_provider.py      # Local Ollama integration
├── tools/
│   └── financial_calculator.py # Financial calculation tools
├── classifiers/
│   └── question_classifier.py  # Question type classification
└── evaluation/
    └── benchmark_runner.py     # FinanceQA evaluation harness
```

### Question Types and Reasoning Strategies

The agent automatically classifies questions into three types:

1. **Tactical-Basic**: Direct calculations with complete data
   - Uses financial tools directly
   - High confidence, precise results
   - Example: "Calculate ROI for $1000 investment with $1200 return"

2. **Tactical-Assumption**: Calculations requiring assumptions
   - Identifies missing data
   - Generates reasonable assumptions
   - Provides confidence intervals
   - Example: "Estimate the NPV of this project" (missing discount rate)

3. **Conceptual**: Theory and explanation-based questions
   - Retrieves financial knowledge
   - Structures explanations logically
   - Provides examples and context
   - Example: "Explain the difference between NPV and IRR"

### Reasoning Chain Inspection

Every question generates a complete reasoning chain:

```python
# Inspect reasoning process
for step in chain.steps:
    print(f"Step: {step.description}")
    print(f"Status: {step.status.value}")
    print(f"Type: {step.step_type.value}")
    if step.error_message:
        print(f"Error: {step.error_message}")
    print("---")
```

## Configuration

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
  "default_provider": "ollama"
}
```

### Agent Configuration

Key parameters in `config/agent_config.py`:

- `temperature`: Model creativity (0.1 for financial precision)
- `max_reasoning_steps`: Limit reasoning chain length
- `tool_timeout`: Individual tool execution timeout
- `confidence_threshold`: Minimum confidence for definitive answers

## Development

### Adding New Financial Tools

1. **Create tool class**:
   ```python
   from src.tools.financial_calculator import FinancialTool

   class MyFinancialTool(FinancialTool):
       @property
       def name(self) -> str:
           return "my_tool"

       async def execute(self, input_data):
           # Implementation here
           return {"result": calculated_value}
   ```

2. **Register with agent**:
   ```python
   agent.add_tool(MyFinancialTool())
   ```

### Testing New Components

1. **Unit tests**: Test individual components in isolation
2. **Integration tests**: Test component interaction
3. **Benchmark tests**: Evaluate on FinanceQA subset
4. **Error cases**: Test failure modes and recovery

### Performance Optimization

Monitor key metrics:

```bash
# Profile agent performance
python scripts/profile_agent.py

# Monitor token usage
python scripts/token_analysis.py

# Benchmark specific question types
python scripts/benchmark_by_type.py
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags

   # Restart Ollama
   ollama serve
   ```

2. **Model Not Found**:
   ```bash
   # List available models
   ollama list

   # Pull required model
   ollama pull llama2
   ```

3. **Import Errors**:
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt

   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

4. **Performance Issues**:
   - Use smaller models for development (phi, mistral)
   - Reduce `max_reasoning_steps`
   - Enable result caching
   - Monitor memory usage

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

## Performance Benchmarks

Run comprehensive benchmarks:

```bash
# Full FinanceQA evaluation (takes ~30 minutes)
python src/evaluation/benchmark_runner.py \
    --evaluation-type full \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Quick evaluation (5 minutes)
python src/evaluation/benchmark_runner.py \
    --evaluation-type quick \
    --num-samples 50 \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Evaluate by question type
python src/evaluation/benchmark_runner.py \
    --evaluation-type by-type \
    --question-type "calculation" \
    --dataset-path data/datasets/financeqa \
    --output-dir results/

# Performance profiling with different splits
python src/evaluation/benchmark_runner.py \
    --evaluation-type full \
    --split validation \
    --dataset-path data/datasets/financeqa \
    --output-dir results/
```

Expected benchmark results:
- **Response Time**: 2-5 seconds per question
- **Memory Usage**: <2GB for Llama2 7B
- **Token Efficiency**: ~1000 tokens per response
- **Tool Success Rate**: >90% for calculation questions

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Run full test suite (`pytest`)
5. Run benchmark evaluation
6. Submit pull request with performance metrics

## License

[License information here]

## Citation

If you use this work in research, please cite:

```bibtex
@software{financeqa_agent,
  title={FinanceQA AI Agent: Multi-step Reasoning for Financial Analysis},
  author={[Authors]},
  year={2024},
  url={[Repository URL]}
}
```