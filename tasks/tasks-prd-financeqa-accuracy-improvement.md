# Task List: FinanceQA Agent Accuracy Improvement System

## Relevant Files

- `src/tools/financial_calculator.py` - Core financial calculator tool requiring enhancement with EBITDA and equity value calculations
- `src/tools/accounting_standards.py` - New file for accounting convention validation (to be created)
- `src/agent/core.py` - Main agent with reasoning chains requiring tool integration and synthesis improvements
- `src/tools/financial_data_extractor.py` - New file for enhanced financial data extraction patterns (to be created)
- `src/tools/assumption_engine.py` - New file for structured assumption generation (to be created)
- `tests/tools/test_financial_calculator.py` - Unit tests for enhanced financial calculator
- `tests/tools/test_accounting_standards.py` - Unit tests for accounting standards validation
- `tests/agent/test_core_integration.py` - Integration tests for improved agent functionality
- `tests/tools/test_assumption_engine.py` - Unit tests for assumption generation engine

### Notes

- Tests should be created alongside implementation to validate accuracy improvements
- Use `pytest` to run tests (this project uses pytest, not jest)
- Focus on low effort, high gain improvements first to maximize immediate accuracy impact

## Tasks

**Priority: LOW EFFORT, HIGH GAIN → MEDIUM EFFORT, HIGH GAIN → HIGH EFFORT, MEDIUM GAIN**

- [ ] 1.0 Fix Critical Financial Calculation Errors (LOW EFFORT, HIGH GAIN - Week 1)
  - [ ] 1.1 Add EBITDA margin calculation method to FinancialCalculator class
  - [ ] 1.2 Implement equity value calculation logic (EV - Net Debt + Cash formula)
  - [ ] 1.3 Fix financial ratio calculations with proper averaging methods
  - [ ] 1.4 Add calculation method detection in `_infer_calculation_type()`
  - [ ] 1.5 Create unit tests for new calculation methods
  - [ ] 1.6 Test against known failing examples (EBITDA 3.65% vs 4.53%, equity value errors)

- [ ] 2.0 Improve Tool Integration and Answer Synthesis (LOW EFFORT, HIGH GAIN - Week 1-2)
  - [ ] 2.1 Enhance `_prepare_tool_input()` method in core.py to detect financial metrics
  - [ ] 2.2 Improve financial data extraction patterns for complex questions
  - [ ] 2.3 Add financial-specific answer formatting in `_synthesize_answer()`
  - [ ] 2.4 Update synthesis prompt to handle currency, percentage, and ratio formatting
  - [ ] 2.5 Add metric-specific data validation before tool execution
  - [ ] 2.6 Create integration tests for improved tool input/output flow

- [ ] 3.0 Add Basic Accounting Standards Validation (MEDIUM EFFORT, HIGH GAIN - Week 2-3)
  - [ ] 3.1 Create new `AccountingStandardsValidator` class in accounting_standards.py
  - [ ] 3.2 Implement EBITDA calculation validation (operating leases, one-time expenses)
  - [ ] 3.3 Add ratio calculation standards validation (average vs end-period)
  - [ ] 3.4 Implement revenue recognition validation (total vs net sales)
  - [ ] 3.5 Integrate validation into FinancialCalculator execution flow
  - [ ] 3.6 Add validation warnings and error messages for non-standard methods
  - [ ] 3.7 Create comprehensive test suite for accounting standards validation

- [ ] 4.0 Enhance Financial Data Extraction (MEDIUM EFFORT, HIGH GAIN - Week 3-4)
  - [ ] 4.1 Create FinancialDataExtractor class in financial_data_extractor.py
  - [ ] 4.2 Implement pattern recognition for financial statements and metrics
  - [ ] 4.3 Add number extraction with context (revenue, EBITDA, shares outstanding)
  - [ ] 4.4 Build financial entity recognition (company names, financial terms)
  - [ ] 4.5 Create data validation and normalization methods
  - [ ] 4.6 Integrate extractor with existing `_extract_financial_data()` in core.py
  - [ ] 4.7 Add comprehensive tests for data extraction accuracy

- [ ] 5.0 Implement Structured Assumption Generation Engine (HIGH EFFORT, HIGH GAIN - Week 4-8)
  - [ ] 5.1 Create AssumptionEngine class in assumption_engine.py
  - [ ] 5.2 Implement financial assumption pattern library (discount rates, growth rates, ratios)
  - [ ] 5.3 Add LLM-enhanced assumption generation with structured prompts
  - [ ] 5.4 Build assumption validation against industry benchmarks
  - [ ] 5.5 Implement assumption confidence scoring and uncertainty quantification
  - [ ] 5.6 Add sensitivity analysis for assumption-dependent calculations
  - [ ] 5.7 Create assumption documentation and reasoning tracking
  - [ ] 5.8 Integrate assumption engine with existing reasoning chain in core.py
  - [ ] 5.9 Update `_generate_assumptions()` and `_identify_missing_data()` methods
  - [ ] 5.10 Create comprehensive test suite for assumption generation scenarios

- [ ] 6.0 Add Advanced Financial Domain Coverage (HIGH EFFORT, MEDIUM GAIN - Week 8-12)
  - [ ] 6.1 Implement M&A analysis calculations (accretion/dilution analysis)
  - [ ] 6.2 Add complex valuation methodologies (multiples analysis, enhanced DCF)
  - [ ] 6.3 Create risk analysis calculations (beta, volatility, VaR)
  - [ ] 6.4 Implement corporate finance metrics (WACC, cost of equity, leverage ratios)
  - [ ] 6.5 Build cross-validation system for multi-method calculation verification
  - [ ] 6.6 Add reasonableness checks and confidence scoring for financial results
  - [ ] 6.7 Create result explanation generation for complex calculations
  - [ ] 6.8 Integrate advanced calculations with existing NPV/IRR framework
  - [ ] 6.9 Add comprehensive test coverage for advanced financial domain calculations
  - [ ] 6.10 Create performance benchmarks and accuracy validation against known correct answers