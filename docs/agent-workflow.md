# FinanceQA Agent Workflow Documentation

## Overview

The FinanceQA agent is a sophisticated multi-step reasoning system designed for financial question answering. It combines large language models with specialized tools and implements adaptive reasoning strategies based on question classification.

## Architecture

### High-Level Components

```
User Question → Agent → ModelManager → OllamaProvider → Local LLM
             ↑                    ↓
    ReasoningChain ← Tools ← Financial Calculators
```

**Core Components:**
- **Agent**: Orchestrates the reasoning process and manages workflow
- **ModelManager**: Handles LLM provider switching (Ollama ↔ OpenAI)
- **ReasoningChain**: Tracks the complete decision process with full observability
- **Tools**: Pluggable financial calculators and utilities
- **OllamaProvider**: Manages local LLM communication with connection pooling

## Main Workflow: 4-Phase Process

### Phase 1: Question Analysis

**Purpose**: Understand the question type, requirements, and scope

**Implementation**:
```python
async def _analyze_question(self, question: str, chain: ReasoningChain) -> ReasoningStep:
    analysis_prompt = f"""
    Analyze this financial question and identify:
    1. Question type (calculation, analysis, conceptual)
    2. Required information and data
    3. Potential tools or methods needed
    4. Expected answer format

    Question: {question}
    """
```

**Output**:
- Question type classification
- Required data identification
- Tool recommendations
- Expected answer format

### Phase 2: Planning

**Purpose**: Generate a structured approach to answer the question

**Implementation**:
```python
async def _generate_reasoning_plan(self, chain: ReasoningChain) -> ReasoningStep:
    plan_prompt = f"""
    Based on the question analysis, create a step-by-step reasoning plan.

    Question: {chain.question}
    Analysis: {chain.steps[-1].output_data.get('analysis', '')}
    Available tools: {list(self.tools.keys())}
    """
```

**Output**:
- Step-by-step reasoning plan
- Tool selection strategy
- Expected workflow path

### Phase 3: Execution (Classification & Routing)

**Purpose**: Execute specialized reasoning based on question type

The agent classifies questions into three categories and routes to appropriate handlers:

#### 🎯 Tactical Basic (Direct Calculations)

**When**: Questions with provided data requiring direct calculations

**Workflow**:
```
Data Extraction → Tool Selection → Calculations → Validation
```

**Steps**:
1. **Data Extraction**: Extract numerical values and financial metrics
2. **Tool Selection**: Choose appropriate calculators (NPV, IRR, ratios)
3. **Calculations**: Execute computations with selected tools
4. **Validation**: Check results for reasonableness

**Example Questions**:
- "Calculate the NPV with cash flows [100, 200, 300] and discount rate 10%"
- "What is the current ratio if current assets are $500K and current liabilities are $200K?"

#### 🤔 Tactical Assumption (Missing Data Calculations)

**When**: Questions requiring calculations but missing critical data

**Workflow**:
```
Missing Data ID → Generate Assumptions → Calculations → Sensitivity Analysis
```

**Steps**:
1. **Missing Data Identification**: Determine what information is absent
2. **Assumption Generation**: Create reasonable assumptions using industry standards
3. **Calculations**: Perform computations with assumed values
4. **Sensitivity Analysis**: Test impact of assumption variations

**Example Questions**:
- "What's the company's ROI?" (missing financial data)
- "Calculate EBITDA for 2024" (missing detailed financial statements)

#### 📚 Conceptual (Explanatory Questions)

**When**: Questions seeking understanding rather than calculations

**Workflow**:
```
Knowledge Retrieval → Structure Explanation → Add Examples
```

**Steps**:
1. **Knowledge Retrieval**: Access financial knowledge base or use LLM knowledge
2. **Structure Explanation**: Organize information logically
3. **Add Examples**: Include practical illustrations

**Example Questions**:
- "What is the difference between NPV and IRR?"
- "Explain the concept of working capital"

### Phase 4: Synthesis

**Purpose**: Combine all reasoning steps into a coherent final answer

**Implementation**:
```python
async def _synthesize_answer(self, chain: ReasoningChain) -> ReasoningStep:
    synthesis_prompt = f"""
    Based on all the reasoning steps, provide a final answer to the question.

    Original Question: {chain.question}
    Reasoning Context: {context}

    Provide a clear, concise final answer.
    """
```

**Output**:
- Final answer combining all intermediate results
- Confidence assessment based on step success rates
- Properly formatted response for question type

## Data Structures

### ReasoningStep

Tracks individual operations within the reasoning process:

```python
@dataclass
class ReasoningStep:
    id: str                           # Unique identifier
    step_type: StepType              # ANALYSIS, CALCULATION, TOOL_USE, etc.
    description: str                 # Human-readable description
    status: StepStatus              # PENDING, IN_PROGRESS, COMPLETED, FAILED
    input_data: Optional[Dict]       # Input parameters
    output_data: Optional[Dict]      # Step results
    tool_used: Optional[str]         # Tool identifier if applicable
    error_message: Optional[str]     # Error details if failed
    confidence_score: Optional[float] # Step confidence (0.0-1.0)
    execution_time_ms: Optional[int] # Performance tracking
    timestamp: datetime              # Execution timestamp
    parent_step_id: Optional[str]    # For hierarchical relationships
    child_step_ids: List[str]        # Child step references
```

**Step Types**:
- `ANALYSIS`: Question analysis, data extraction, knowledge retrieval
- `CALCULATION`: Financial computations and tool execution
- `TOOL_USE`: External tool selection and preparation
- `CLASSIFICATION`: Question type determination
- `ASSUMPTION`: Missing data identification and assumption generation
- `SYNTHESIS`: Answer compilation and structuring
- `VALIDATION`: Result verification and sanity checking

**Step Statuses**:
- `PENDING`: Not yet started
- `IN_PROGRESS`: Currently executing
- `COMPLETED`: Successfully finished
- `FAILED`: Encountered error
- `SKIPPED`: Intentionally bypassed

### ReasoningChain

Container for the complete reasoning process:

```python
@dataclass
class ReasoningChain:
    id: str                          # Unique chain identifier
    question: str                    # Original question
    steps: List[ReasoningStep]       # Ordered list of reasoning steps
    final_answer: Optional[str]      # Synthesized answer
    confidence_score: Optional[float] # Overall confidence
    total_execution_time_ms: Optional[int] # Total processing time
    metadata: Dict[str, Any]         # Additional context
    created_at: datetime             # Creation timestamp
    completed_at: Optional[datetime] # Completion timestamp
```

**Key Methods**:
- `add_step()`: Append new reasoning step
- `get_step_by_id()`: Retrieve specific step
- `get_steps_by_type()`: Filter steps by type
- `get_failed_steps()`: Identify error steps
- `is_complete()`: Check completion status
- `has_failures()`: Detect any failures
- `to_dict()`: Serialize for storage/analysis

## Tool Architecture

### Tool Interface

All tools implement a standardized async interface:

```python
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for selection."""

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with input data."""

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate tool input data."""
```

### Available Tools

**Financial Calculator**:
- **Purpose**: NPV, IRR, ROI, financial ratio calculations
- **Keywords**: "npv", "irr", "roi", "return", "ratio", "calculate"
- **Input**: Numerical data, calculation type, parameters
- **Output**: Computed results with metadata

**Document Parser**:
- **Purpose**: Extract financial data from statements and documents
- **Keywords**: "statement", "document", "extract", "parse"
- **Input**: Text or document content
- **Output**: Structured financial data

**Code Executor**:
- **Purpose**: Complex multi-step analysis requiring programming
- **Keywords**: "complex", "multiple", "steps", "analysis"
- **Input**: Problem description, data, language preference
- **Output**: Executed code results

**Knowledge Base**:
- **Purpose**: Retrieve relevant financial knowledge (RAG system)
- **Keywords**: Conceptual questions
- **Input**: Query text, retrieval type
- **Output**: Relevant knowledge snippets

### Tool Selection Logic

```python
tool_keywords = {
    "financial_calculator": ["npv", "irr", "roi", "return", "calculate", "ratio"],
    "code_executor": ["complex", "multiple", "steps", "analysis"],
    "document_parser": ["statement", "document", "extract", "parse"]
}

# Select tools based on question keywords
for tool_name, keywords in tool_keywords.items():
    if tool_name in self.tools and any(kw in question_lower for kw in keywords):
        selected_tools.append(tool_name)
```

## Safety & Error Handling

### Infinite Loop Prevention

**Max Steps Limit**: Default 15 steps with checks throughout workflow
```python
if len(chain.steps) >= self.max_steps:
    logger.warning(f"Reached max steps limit ({self.max_steps}), stopping early")
    return chain
```

**Timeouts**: 60-second timeout on each LLM request
```python
response = await asyncio.wait_for(
    self.model_manager.generate(prompt),
    timeout=60
)
```

**Session Management**: Connection pooling with limits
```python
connector = aiohttp.TCPConnector(
    limit=10,  # Max 10 concurrent connections
    limit_per_host=5,  # Max 5 per host
)
```

**Concurrency Control**: Max 3 concurrent evaluations in benchmark mode
```python
semaphore = asyncio.Semaphore(3)  # Max 3 concurrent evaluations
```

### Error Recovery

**Safe Step Execution**:
```python
async def _safe_execute_step(self, step_func, *args, **kwargs) -> ReasoningStep:
    try:
        return await step_func(*args, **kwargs)
    except Exception as e:
        # Create failed step with error message
        # Continue with degraded functionality
```

**LLM Fallback**:
```python
async def _safe_model_generate(self, prompt: str, fallback_message: str = None, timeout: int = 60) -> str:
    try:
        response = await asyncio.wait_for(self.model_manager.generate(prompt), timeout=timeout)
        return response.content
    except asyncio.TimeoutError:
        return fallback_message or "Request timed out."
    except Exception as e:
        return fallback_message or "Unable to generate response due to technical issues."
```

**Tool Fallback**:
```python
async def _safe_tool_execute(self, tool_name: str, tool_input: Dict[str, Any], question: str) -> Dict[str, Any]:
    try:
        result = await self.tools[tool_name].execute(tool_input)
        return {"status": "success", "result": result, "tool": tool_name}
    except Exception as e:
        # Fall back to LLM-only response
        llm_response = await self._safe_model_generate(fallback_prompt)
        return {"status": "llm_fallback", "result": {"response": llm_response}, "tool": "llm"}
```

**Graceful Degradation Principles**:
- Failed tools fall back to LLM-only responses
- Empty LLM responses trigger fallback messages
- Step failures are logged but don't stop the chain
- Final answer is always ensured, even if incomplete

## Model Management

### Provider Architecture

**ModelManager**: Handles multiple LLM providers with seamless switching
```python
class ModelManager:
    def __init__(self):
        self._providers: Dict[str, ModelProvider] = {}
        self._default_provider: Optional[str] = None

    async def generate(self, prompt: str, provider_name: Optional[str] = None) -> ModelResponse:
        provider = self.get_provider(provider_name)
        return await provider.generate(prompt)
```

**OllamaProvider**: Local LLM integration with enhanced reliability
```python
class OllamaProvider(ModelProvider):
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        # Enhanced session management with connection pooling
        # Timeout handling and empty response detection
        # Proper error recovery and logging
```

**Configuration**: Model switching via JSON configuration
```json
{
  "providers": {
    "ollama": {
      "type": "ollama",
      "model_name": "mistral:latest",
      "base_url": "http://localhost:11435",
      "temperature": 0.1,
      "max_tokens": 12000,
      "timeout": 600
    }
  },
  "default_provider": "ollama"
}
```

## Performance & Monitoring

### Execution Tracking

**Step-Level Metrics**:
- Execution time per step
- Tool usage statistics
- Error rates and types
- Confidence scores

**Chain-Level Metrics**:
- Total processing time
- Step count and distribution
- Success/failure rates
- Question type patterns

**System-Level Metrics**:
- Concurrent request handling
- Resource utilization
- Provider availability
- Session management

### Benchmark Evaluation

**Evaluation Pipeline**:
```python
# Concurrency controls for benchmark evaluation
semaphore = asyncio.Semaphore(3)  # Max 3 concurrent evaluations
timeout = 300.0  # 5-minute timeout per evaluation

async def evaluate_with_timeout(example):
    return await asyncio.wait_for(evaluate_with_semaphore(example), timeout=300.0)
```

**Features**:
- **Parallel Processing**: Multiple questions processed concurrently
- **Timeout Protection**: Individual and total evaluation timeouts
- **Progress Tracking**: Real-time progress bars and metrics
- **Result Persistence**: Automatic saving of evaluation results
- **Failure Handling**: Graceful degradation for failed evaluations

### Observability

**Serializable Chains**: Full reasoning processes can be saved to JSON
```json
{
  "id": "uuid",
  "question": "What is NPV?",
  "steps": [...],
  "final_answer": "Net Present Value is...",
  "confidence_score": 0.85,
  "total_execution_time_ms": 2340,
  "metadata": {...}
}
```

**Logging**: Comprehensive logging at all levels
- Step execution and failures
- Tool selection and results
- Model interactions and timeouts
- Performance metrics

## Usage Examples

### Basic Usage

```python
from src.models.model_manager import create_model_manager_from_config
from src.agent.core import Agent

# Load configuration
config = {...}  # Load from JSON
model_manager = create_model_manager_from_config(config)

# Create agent with safety limits
agent = Agent(model_manager, tools=[], max_steps=15)

# Answer question
chain = await agent.answer_question("What is the NPV of cash flows [100, 200, 300] at 10% discount rate?")

# Access results
print(f"Answer: {chain.final_answer}")
print(f"Steps: {len(chain.steps)}")
print(f"Confidence: {chain.confidence_score}")
```

### Advanced Usage with Tools

```python
from src.tools.financial_calculator import FinancialCalculator

# Create tools
financial_calc = FinancialCalculator()

# Create agent with tools
agent = Agent(model_manager, tools=[financial_calc], max_steps=20)

# Complex question requiring tools
chain = await agent.answer_question("Calculate the IRR for an investment...")

# Analyze reasoning process
for step in chain.steps:
    print(f"{step.description}: {step.status.value}")
    if step.tool_used:
        print(f"  Tool: {step.tool_used}")
    if step.execution_time_ms:
        print(f"  Time: {step.execution_time_ms}ms")
```

### Evaluation and Benchmarking

```python
from src.evaluation.benchmark_runner import FinanceQAEvaluator

# Create evaluator
evaluator = FinanceQAEvaluator()

# Run evaluation with concurrency controls
results = await evaluator.evaluate_agent_async(
    agent=agent,
    max_examples=50,
    verbose=True
)

# Analyze results
print(f"Accuracy: {results['metrics']['exact_match_accuracy']:.1%}")
print(f"Average time: {results['average_processing_time']:.1f}s")
```

## Design Principles

1. **Modularity**: Each reasoning path is independent and specialized
2. **Observability**: Every step is tracked and serializable for analysis
3. **Resilience**: Multiple fallback mechanisms and error recovery strategies
4. **Performance**: Concurrent processing with appropriate resource limits
5. **Extensibility**: Pluggable tools and provider architecture
6. **Safety**: Comprehensive timeout and loop prevention mechanisms

## Configuration

### Agent Configuration

```python
# Agent initialization parameters
agent = Agent(
    model_manager=model_manager,
    tools=tool_list,              # Optional list of available tools
    max_steps=15                  # Maximum reasoning steps (default: 15)
)
```

### Model Configuration

```json
{
  "providers": {
    "ollama": {
      "type": "ollama",
      "model_name": "mistral:latest",
      "base_url": "http://localhost:11435",
      "temperature": 0.1,           # Lower for financial reasoning
      "max_tokens": 12000,
      "timeout": 600               # Request timeout in seconds
    }
  },
  "default_provider": "ollama"
}
```

### Evaluation Configuration

```python
# Benchmark evaluation settings
evaluation_settings = {
    "max_concurrent": 3,          # Concurrent evaluations
    "timeout_per_evaluation": 300, # 5-minute timeout
    "max_examples": 50,           # Evaluation subset size
    "verbose": True               # Progress display
}
```

This architecture enables the FinanceQA agent to handle complex financial questions through systematic reasoning while maintaining reliability and performance at scale.