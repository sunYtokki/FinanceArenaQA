# PRD: FinanceQA Agent Accuracy Improvement System

## Introduction/Overview

The FinanceQA Agent currently achieves ~40% overall accuracy with critical failures in assumption-based questions (2% accuracy) and basic financial calculations (EBITDA margin off by 25%, equity value off by 17x). This PRD addresses the comprehensive accuracy improvement system through iterative 30-day cycles, focusing on core calculation and logic fixes while maintaining the existing robust multi-step reasoning architecture.

**Problem Statement**: Despite having sophisticated reasoning chains and financial tools, the agent fails on fundamental financial calculations due to missing domain-specific logic, inadequate accounting convention validation, and poor assumption generation capabilities.

**Goal**: Systematically improve accuracy from ~40% to 70%+ overall, with assumption questions improving from 2% to 20%+ accuracy through targeted fixes to financial calculation logic gaps.

## Goals

1. **Phase 1 (30 days)**: Fix critical calculation errors - achieve 55-60% overall accuracy
2. **Phase 2 (60 days)**: Implement assumption generation engine - achieve 15%+ assumption question accuracy
3. **Phase 3 (90 days)**: Complete financial domain coverage - achieve 70%+ overall accuracy
4. **Cross-Phase**: Maintain/improve existing strengths (NPV, IRR calculations already excellent)

## User Stories

### **Financial Analyst User**
- **As a** financial analyst, **I want** accurate EBITDA margin calculations **so that** I can trust the agent's basic financial metrics
- **As a** financial analyst, **I want** proper equity value calculations **so that** I can use the agent for valuation work
- **As a** financial analyst, **I want** assumption-based calculations with clear reasoning **so that** I can understand and validate the agent's logic

### **Investment Professional User**
- **As an** investment professional, **I want** calculations that follow accounting standards **so that** results match my manual calculations
- **As an** investment professional, **I want** sensitivity analysis for assumption-based questions **so that** I can assess calculation reliability

### **Developer/Maintainer User**
- **As a** developer, **I want** modular financial calculation components **so that** I can easily add new financial metrics
- **As a** developer, **I want** comprehensive test coverage **so that** I can validate accuracy improvements

## Functional Requirements

### **Phase 1: Core Calculation Logic Fixes (Days 1-30)**

#### **FR1: Enhanced Financial Calculator**
1.1. Add specific EBITDA calculation methods with accounting standard compliance
1.2. Implement equity value calculation logic (EV - Net Debt + Cash formula)
1.3. Add financial ratio calculations with proper averaging methods (AP Days using average AP)
1.4. Integrate accounting convention validation directly into FinancialCalculator class
1.5. Support multiple calculation methodologies (GAAP vs IFRS standards)

#### **FR2: Accounting Standards Validation**
2.1. Validate EBITDA calculations ensure proper adjustments (operating leases, one-time expenses)
2.2. Enforce ratio calculation standards (average balance sheet items vs end-period)
2.3. Verify revenue recognition (total revenue vs net sales for margin calculations)
2.4. Flag non-standard calculation methods with warnings

#### **FR3: Improved Tool Integration**
3.1. Enhance `_prepare_tool_input()` method to detect specific financial metrics
3.2. Implement financial data extraction patterns for complex questions
3.3. Add metric-specific data validation before calculation
3.4. Improve answer synthesis with financial-specific formatting rules

### **Phase 2: Assumption Generation Engine (Days 31-60)**

#### **FR4: LLM-Enhanced Assumption System**
4.1. Implement structured assumption generation prompts for common financial scenarios
4.2. Create assumption validation logic against industry standards/benchmarks
4.3. Build assumption confidence scoring based on data availability
4.4. Add sensitivity analysis for assumption-dependent calculations
4.5. Document assumption reasoning in calculation results

#### **FR5: Financial Context Understanding**
5.1. Detect missing data patterns in financial questions
5.2. Generate reasonable assumptions based on financial domain knowledge
5.3. Provide assumption alternatives with confidence ranges
5.4. Integrate assumption generation with existing reasoning chain

### **Phase 3: Comprehensive Financial Domain Coverage (Days 61-90)**

#### **FR6: Advanced Financial Calculations**
6.1. M&A analysis calculations (accretion/dilution analysis)
6.2. Complex valuation methodologies (multiples analysis, DCF enhancements)
6.3. Risk analysis calculations (beta, volatility, VaR)
6.4. Corporate finance metrics (WACC, cost of equity, leverage ratios)

#### **FR7: Cross-Validation System**
7.1. Implement multi-method calculation verification
7.2. Add reasonableness checks for financial results
7.3. Create confidence scoring based on calculation method consistency
7.4. Build result explanation generation for complex calculations

## Non-Goals (Out of Scope)

1. **Model Architecture Changes**: Maintain existing ReasoningChain and Agent framework
2. **New Financial Data Sources**: Use existing document parsing and data extraction
3. **UI/Interface Changes**: Focus purely on calculation accuracy logic
4. **Performance Optimization**: Maintain current speed, focus on accuracy
5. **New Question Types**: Improve existing tactical/conceptual/assumption categories
6. **Evaluation Infrastructure**: Maintain existing benchmark runner setup

## Technical Considerations

### **Integration Points**
- **FinancialCalculator Enhancement**: Extend existing `src/tools/financial_calculator.py` (lines 35-706)
- **Core Agent Integration**: Enhance `_prepare_tool_input()` and `_synthesize_answer()` in `src/agent/core.py`
- **Classification System**: Leverage existing QuestionClassifier for routing improved logic

### **Architecture Constraints**
- Must maintain async tool execution pattern
- Preserve existing error handling and fallback mechanisms
- Keep ReasoningStep tracking for transparency
- Maintain model-agnostic design (Ollama/OpenAI compatibility)

### **Dependencies**
- Existing model_manager for LLM-enhanced assumptions
- Current tool execution framework
- Established question classification system

## Success Metrics

### **Accuracy Targets**
- **Overall Accuracy**: 40% → 60% (Phase 1) → 65% (Phase 2) → 70%+ (Phase 3)
- **Tactical Basic Questions**: 42% → 65% → 70% → 75%
- **Tactical Assumption Questions**: 2% → 8% → 15% → 20%+
- **Conceptual Questions**: Maintain 64%+ (already strong)

### **Specific Calculation Accuracy**
- **EBITDA Margin Calculations**: Fix 25% error gap → <5% error rate
- **Equity Value Calculations**: Fix 17x error → <10% error rate
- **Financial Ratio Calculations**: Implement proper averaging → 90%+ accuracy

### **Quality Metrics**
- **Assumption Generation**: 80%+ reasonable assumptions for missing data scenarios
- **Accounting Compliance**: 95%+ calculations follow standard conventions
- **Error Reduction**: 50% reduction in calculation-related failures

## Open Questions

1. **Assumption Confidence Thresholds**: What confidence levels should trigger human review vs automatic processing?
2. **Accounting Standards Priority**: Should GAAP take precedence over IFRS when standards conflict?
3. **Calculation Method Selection**: How should the system choose between multiple valid calculation approaches?
4. **Performance Impact**: What's the acceptable latency increase for enhanced accuracy?
5. **Error Handling Granularity**: How detailed should calculation error messages be for debugging?

## Implementation Logic Gaps Addressed

### **Current Logic Gap Analysis**

#### **Gap 1: Missing Financial Domain Logic**
- **Current**: Generic calculation methods in `financial_calculator.py`
- **Fix**: Add financial metric-specific calculation logic with accounting standards
- **Impact**: Fixes EBITDA margin and equity value calculation errors

#### **Gap 2: Poor Tool Input Preparation**
- **Current**: `_prepare_tool_input()` uses simplistic data extraction (lines 618-641)
- **Fix**: Financial metric detection and structured data preparation
- **Impact**: Improves calculator tool effectiveness

#### **Gap 3: Weak Assumption Generation**
- **Current**: Generic LLM prompts for missing data (lines 763-806)
- **Fix**: Structured financial assumption patterns with validation
- **Impact**: Addresses 2% assumption question accuracy crisis

#### **Gap 4: Answer Synthesis Issues**
- **Current**: Generic synthesis prompt doesn't handle financial formatting
- **Fix**: Financial-specific answer formatting and validation
- **Impact**: Fixes final answer extraction problems

#### **Gap 5: No Accounting Convention Enforcement**
- **Current**: No validation of calculation methods against standards
- **Fix**: Integrated accounting standards validation
- **Impact**: Ensures calculation methodology correctness

This PRD systematically addresses each identified logic gap through targeted functional requirements while maintaining the existing architectural strengths.