"""Agent prompts for all reasoning chain steps in the financial agent."""


# Step 1: Analysis prompts
ENHANCED_ANALYSIS_PROMPT = """
Analyze this financial question to identify required concepts and data for retrieval.

Question: {question}

## Your Task

Based on the question, determine:
1. What financial concepts are involved?
2. What contextual data or information is needed?
3. What search keywords would find similar questions and reasoning in the Q&A dataset?
4. Which data sources should be consulted?

## Source Routing Logic

**Use ["10k"]** when:
- Question explicitly mentions "Costco" or references "the company" with provided context
- Asks for specific company financial metrics, operational data, or business information
- Examples: "What is Costco's gross margin?", "How many warehouses does Costco operate?"

**Use ["qa_dataset"]** when:
- Question asks "what is", "explain", "how to calculate", or "define"
- Focuses on general financial concepts without specific company context
- Examples: "What is working capital?", "How do you calculate ROE?", "Explain inventory turnover"

**Use ["both"]** when:
- Question requires company-specific data AND financial theory/methodology
- Involves estimation, projection, comparison, or analysis requiring both sources
- Company is mentioned but question needs conceptual understanding
- Examples: "Is Costco's P/E ratio healthy?", "Calculate Costco's working capital", "Analyze Costco's liquidity"

## Instructions

1. **financial_concepts**: List 2-5 key financial terms/metrics relevant to answering this question
   - Use standard financial terminology (e.g., "EBITDA", "working capital", "debt-to-equity")
   
2. **required_context**: Identify 1-3 essential pieces of information needed
   - Be specific: "balance sheet data", "income statement figures", "theoretical framework", etc.
   
3. **search_keywords**: Generate 3-7 keywords to find similar questions in the Q&A dataset
   - Include the main financial concept, calculation method, and related terms
   - Optimize for semantic similarity (e.g., "calculate inventory turnover ratio", "days inventory outstanding")
   
4. **recommended_sources**: Single-element array based on routing logic above

## Output Format

Provide ONLY a single JSON object with the required fields.
Wrap the JSON with <output> ... </output>.
No other text, tags, or code fences.

Required JSON fields:
- financial_concepts: string[]
- required_context: string[]
- search_keywords: string[]
- recommended_sources: ["10k"] | ["qa_dataset"] | ["both"]

Start your response exactly like this:
<output>{{
  "financial_concepts": ["...", "..."],
  "required_context": ["..."],
  "search_keywords": ["...", "..."],
  "recommended_sources": ["..."]
}}</output>

## Examples

Question: "What is EBITDA?"
<output>{{
  "financial_concepts": ["EBITDA", "earnings", "operating performance"],
  "required_context": ["definition", "calculation methodology"],
  "search_keywords": ["EBITDA definition", "earnings before interest taxes depreciation", "EBITDA calculation"],
  "recommended_sources": ["qa_dataset"]
}}</output>

Question: "What was Costco's revenue in 2024?"
<output>{{
  "financial_concepts": ["revenue", "net sales", "total revenue"],
  "required_context": ["Costco 10-K report", "fiscal year 2024 financial statements"],
  "search_keywords": ["revenue calculation", "net sales", "income statement"],
  "recommended_sources": ["10k"]
}}</output>

Question: "Calculate Costco's inventory turnover for 2024"
<output>{{
  "financial_concepts": ["inventory turnover", "cost of goods sold", "average inventory"],
  "required_context": ["Costco financial statements", "inventory turnover formula"],
  "search_keywords": ["calculate inventory turnover ratio", "inventory turnover formula", "COGS divided by average inventory"],
  "recommended_sources": ["both"]
}}</output>
"""


# Step 2: Context/RAG prompts
CONTEXT_BUILDER_PROMPT = """
You are a financial reasoning expert. Extract solution patterns from similar questions and apply them to answer the current question.

**Question:** {question}

**Available Data:**
{provided_context}

**Similar Questions & Solutions:**
{search_results}

## Instructions

1. **Extract Reasoning Patterns**: Identify how similar questions were solved step-by-step
2. **Map to Current Question**: Adapt the reasoning approach to fit the current question
3. **Connect to Available Data**: Link the reasoning steps to specific data in the provided context
4. **Provide Clear Framework**: Create a structured guide for solving the question

## Output Format

**Reasoning Pattern from Similar Questions:**
- [Step-by-step approach that worked for similar problems]

**Applying to Current Question:**
- [How this reasoning applies here, using the available context data]

**Required Calculations/Analysis:**
- [Specific formulas, metrics, or analysis methods needed]

**Data Mapping:**
- [Which data points from context correspond to which parts of the solution]

**Solution Steps:**
1. [First step with data references]
2. [Second step with calculations]
3. [Final step to reach answer]

Begin your analysis:
"""


# Synthesis prompts

SYNTHESIS_PROMPT = """
You are a financial expert answering a specific question. Use the analysis and similar examples to provide a direct, accurate answer.

Question: {question}

Question Analysis: {analysis}

Question Context: {question_context}

Similar Questions and Reasoning Context: {reasoning_context}

Instructions:
1. Answer the question directly based on your financial expertise
2. Use the similar examples and reasoning patterns as guidance when relevant
3. If calculations are needed, perform them step by step
4. If the question requires specific data not provided, state reasonable assumptions
5. Be precise and concise in your answer
6. End with the exact final answer on a new line starting with '<final_answer>:'

Provide your response:
<final_answer>
"""


# Utility functions to get formatted prompts
def get_enhanced_analysis_prompt(question: str) -> str:
    """Get the enhanced analysis prompt for a financial question."""
    return ENHANCED_ANALYSIS_PROMPT.format(question=question)

def get_synthesis_prompt(question: str, analysis: str, question_context: str, reasoning_context: str) -> str:
    """Get the synthesis prompt."""
    return SYNTHESIS_PROMPT.format(
        question=question,
        analysis=analysis,
        question_context=question_context,
        reasoning_context=reasoning_context
    )

def get_context_builder_prompt(question: str, provided_context: str, search_results: list) -> str:
    """Get the context builder prompt with search results."""
    # Format search results for the prompt
    formatted_results = []
    for i, result in enumerate(search_results, 1):
        source_type = result.get('source_type', 'unknown')
        text = result.get('text', '')
        metadata = result.get('metadata', {})
        score = result.get('score', 0.0)

        # Create source attribution
        if source_type == '10k':
            attribution = f"[10-K: {metadata.get('company', 'Unknown')}, {metadata.get('section', 'Unknown')}, Score: {score:.2f}]"
        elif source_type == 'qa_dataset':
            topics = metadata.get('topics', '').split(',') if metadata.get('topics') else []
            attribution = f"[Q&A Dataset: Topics: {', '.join(topics[:2])}, Score: {score:.2f}]"
        else:
            attribution = f"[Source: {source_type}, Score: {score:.2f}]"

        formatted_results.append(f"{i}. {attribution}\n{text}\n")

    search_results_text = "\n".join(formatted_results)

    prompt = CONTEXT_BUILDER_PROMPT.format(
        question=question,
        provided_context=provided_context,
        search_results=search_results_text
    )
    return prompt