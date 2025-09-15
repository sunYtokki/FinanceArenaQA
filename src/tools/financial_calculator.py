"""Simple financial calculation tools."""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FinancialTool(ABC):
    """Base interface for financial calculation tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for selection."""
        pass

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the financial calculation."""
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data - override in subclasses."""
        return isinstance(input_data, dict)


class FinancialCalculator(FinancialTool):
    """Simple financial calculator for basic calculations."""

    @property
    def name(self) -> str:
        return "financial_calculator"

    @property
    def description(self) -> str:
        return "Basic financial calculations (ratios, percentages, ROI)"

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute financial calculations based on operation type."""

        try:
            operation = input_data.get("operation", "general")
            question = input_data.get("question", "")
            data = input_data.get("data", {})

            # Extract numbers from question if not in data
            if not data.get("numbers"):
                import re
                numbers = re.findall(r'\d+\.?\d*', question)
                numbers = [float(n) for n in numbers]
            else:
                numbers = data.get("numbers", [])

            result = {}

            if operation == "roi" or "roi" in question.lower():
                result = self._calculate_roi(numbers, question)
            elif operation == "ratio" or any(term in question.lower() for term in ["ratio", "current ratio", "debt"]):
                result = self._calculate_ratio(numbers, question)
            elif operation == "npv" or "npv" in question.lower():
                result = self._calculate_npv(numbers, question)
            elif operation == "irr" or "irr" in question.lower():
                result = self._calculate_irr(numbers, question)
            else:
                result = self._general_calculation(numbers, question)

            result["status"] = "success"
            result["tool"] = self.name
            return result

        except Exception as e:
            logger.error(f"Financial calculation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tool": self.name
            }

    def _calculate_roi(self, numbers: list, question: str) -> Dict[str, Any]:
        """Calculate Return on Investment."""
        if len(numbers) < 2:
            return {"error": "Need at least 2 numbers for ROI calculation"}

        # Simple ROI = (Gain - Cost) / Cost * 100
        if len(numbers) == 2:
            gain, cost = numbers[0], numbers[1]
            roi = ((gain - cost) / cost) * 100 if cost != 0 else 0

            return {
                "calculation": "ROI",
                "formula": "(Gain - Cost) / Cost * 100",
                "inputs": {"gain": gain, "cost": cost},
                "result": roi,
                "unit": "percentage"
            }

        return {"error": "ROI calculation requires exactly 2 numbers"}

    def _calculate_ratio(self, numbers: list, question: str) -> Dict[str, Any]:
        """Calculate financial ratios."""
        if len(numbers) < 2:
            return {"error": "Need at least 2 numbers for ratio calculation"}

        ratio = numbers[0] / numbers[1] if numbers[1] != 0 else 0

        # Determine ratio type from question
        ratio_type = "general"
        if "current ratio" in question.lower():
            ratio_type = "current_ratio"
        elif "debt" in question.lower():
            ratio_type = "debt_ratio"
        elif "pe ratio" in question.lower() or "price to earnings" in question.lower():
            ratio_type = "pe_ratio"

        return {
            "calculation": f"{ratio_type}_ratio",
            "formula": f"{numbers[0]} / {numbers[1]}",
            "inputs": {"numerator": numbers[0], "denominator": numbers[1]},
            "result": ratio,
            "interpretation": self._interpret_ratio(ratio, ratio_type)
        }

    def _calculate_npv(self, numbers: list, question: str) -> Dict[str, Any]:
        """Enhanced NPV calculation with cash flow analysis and assumptions tracking."""
        if len(numbers) < 2:
            return {"error": "Need initial investment and cash flows for NPV"}

        # Extract discount rate from question or use default
        discount_rate = self._extract_discount_rate(question)
        initial_investment = abs(numbers[0])  # Make positive for calculation
        cash_flows = numbers[1:]

        # Track assumptions made
        assumptions = []

        if "discount" not in question.lower() and "rate" not in question.lower():
            assumptions.append(f"Assumed discount rate of {discount_rate*100:.1f}% (market average)")

        # Perform NPV calculation with detailed tracking
        npv_calculation = self._perform_npv_calculation(initial_investment, cash_flows, discount_rate)

        # Analyze cash flow patterns
        cash_flow_analysis = self._analyze_cash_flows(cash_flows)

        # Calculate additional metrics
        profitability_index = (npv_calculation['present_value_of_cash_flows'] / initial_investment) if initial_investment > 0 else 0
        payback_period = self._calculate_payback_period(initial_investment, cash_flows)

        # Risk assessment
        risk_factors = self._assess_npv_risks(cash_flows, discount_rate)

        result = {
            "calculation": "Enhanced NPV Analysis",
            "formula": "NPV = Σ(CFt / (1 + r)^t) - Initial Investment",
            "inputs": {
                "initial_investment": initial_investment,
                "cash_flows": cash_flows,
                "discount_rate": discount_rate,
                "number_of_periods": len(cash_flows)
            },
            "npv_result": npv_calculation['npv'],
            "present_value_of_cash_flows": npv_calculation['present_value_of_cash_flows'],
            "detailed_calculation": npv_calculation['year_by_year'],
            "profitability_index": profitability_index,
            "payback_period": payback_period,
            "cash_flow_analysis": cash_flow_analysis,
            "assumptions": assumptions,
            "risk_assessment": risk_factors,
            "interpretation": self._interpret_npv_result(npv_calculation['npv'], profitability_index, risk_factors)
        }

        return result

    def _extract_discount_rate(self, question: str) -> float:
        """Extract discount rate from question or return default."""
        import re

        # Look for percentage patterns
        rate_pattern = r'(\d+\.?\d*)%'
        rate_matches = re.findall(rate_pattern, question.lower())

        if rate_matches:
            return float(rate_matches[0]) / 100

        # Look for decimal patterns with rate keywords
        decimal_pattern = r'(?:rate|discount|wacc).*?(\d+\.?\d*)'
        decimal_matches = re.findall(decimal_pattern, question.lower())

        if decimal_matches:
            rate = float(decimal_matches[0])
            # Convert to decimal if it seems to be a percentage
            if rate > 1:
                rate = rate / 100
            return rate

        # Default market rate
        return 0.10

    def _perform_npv_calculation(self, initial_investment: float, cash_flows: list, discount_rate: float) -> Dict[str, Any]:
        """Perform detailed NPV calculation with year-by-year breakdown."""
        year_by_year = []
        total_pv = 0

        for year, cash_flow in enumerate(cash_flows, 1):
            discount_factor = 1 / ((1 + discount_rate) ** year)
            present_value = cash_flow * discount_factor
            total_pv += present_value

            year_by_year.append({
                "year": year,
                "cash_flow": cash_flow,
                "discount_factor": discount_factor,
                "present_value": present_value,
                "cumulative_pv": total_pv
            })

        npv = total_pv - initial_investment

        return {
            "npv": npv,
            "present_value_of_cash_flows": total_pv,
            "year_by_year": year_by_year
        }

    def _analyze_cash_flows(self, cash_flows: list) -> Dict[str, Any]:
        """Analyze cash flow patterns and characteristics."""
        if not cash_flows:
            return {"error": "No cash flows to analyze"}

        analysis = {
            "total_cash_flows": sum(cash_flows),
            "average_annual_cash_flow": sum(cash_flows) / len(cash_flows),
            "cash_flow_growth": [],
            "volatility": 0,
            "pattern": "irregular"
        }

        # Calculate year-over-year growth rates
        for i in range(1, len(cash_flows)):
            if cash_flows[i-1] != 0:
                growth = ((cash_flows[i] - cash_flows[i-1]) / abs(cash_flows[i-1])) * 100
                analysis["cash_flow_growth"].append(growth)

        # Calculate volatility (standard deviation of cash flows)
        if len(cash_flows) > 1:
            mean = analysis["average_annual_cash_flow"]
            variance = sum((cf - mean) ** 2 for cf in cash_flows) / len(cash_flows)
            analysis["volatility"] = variance ** 0.5

        # Determine pattern
        if all(cf > 0 for cf in cash_flows):
            if len(analysis["cash_flow_growth"]) > 0:
                avg_growth = sum(analysis["cash_flow_growth"]) / len(analysis["cash_flow_growth"])
                if avg_growth > 5:
                    analysis["pattern"] = "growing"
                elif avg_growth < -5:
                    analysis["pattern"] = "declining"
                else:
                    analysis["pattern"] = "stable"
            else:
                analysis["pattern"] = "single_period"
        else:
            analysis["pattern"] = "mixed"

        return analysis

    def _calculate_payback_period(self, initial_investment: float, cash_flows: list) -> Optional[float]:
        """Calculate payback period for the investment."""
        cumulative_cash_flow = 0

        for year, cash_flow in enumerate(cash_flows, 1):
            cumulative_cash_flow += cash_flow
            if cumulative_cash_flow >= initial_investment:
                # Linear interpolation for more precise payback period
                if year == 1:
                    return year - (cumulative_cash_flow - initial_investment) / cash_flow
                else:
                    prev_cumulative = cumulative_cash_flow - cash_flow
                    return year - 1 + (initial_investment - prev_cumulative) / cash_flow

        return None  # Payback not achieved within the given period

    def _assess_npv_risks(self, cash_flows: list, discount_rate: float) -> Dict[str, Any]:
        """Assess risks associated with the NPV calculation."""
        risks = {
            "discount_rate_sensitivity": "medium",
            "cash_flow_uncertainty": "medium",
            "key_risks": []
        }

        # Discount rate sensitivity
        if discount_rate > 0.15:
            risks["discount_rate_sensitivity"] = "high"
            risks["key_risks"].append("High discount rate increases sensitivity to rate changes")
        elif discount_rate < 0.05:
            risks["discount_rate_sensitivity"] = "low"
            risks["key_risks"].append("Low discount rate - consider if realistic for project risk")

        # Cash flow uncertainty assessment
        if len(cash_flows) > 5:
            risks["cash_flow_uncertainty"] = "high"
            risks["key_risks"].append("Long-term projections have higher uncertainty")

        # Check for negative cash flows
        negative_flows = [cf for cf in cash_flows if cf < 0]
        if negative_flows:
            risks["key_risks"].append(f"Contains {len(negative_flows)} negative cash flows")

        # Check for very large later cash flows
        if len(cash_flows) > 1:
            early_avg = sum(cash_flows[:2]) / 2 if len(cash_flows) >= 2 else cash_flows[0]
            late_flows = cash_flows[2:] if len(cash_flows) > 2 else []
            if late_flows and any(cf > early_avg * 3 for cf in late_flows):
                risks["key_risks"].append("Large later cash flows increase projection risk")

        return risks

    def _interpret_npv_result(self, npv: float, profitability_index: float, risk_factors: Dict[str, Any]) -> str:
        """Provide comprehensive interpretation of NPV results."""
        if npv > 0:
            interpretation = f"Positive NPV of ${npv:,.2f} indicates value-creating investment. "

            if profitability_index > 1.5:
                interpretation += "High profitability index suggests strong returns. "
            elif profitability_index > 1.2:
                interpretation += "Good profitability index indicates solid returns. "
            else:
                interpretation += "Modest profitability index - acceptable but not exceptional. "

        elif npv < 0:
            interpretation = f"Negative NPV of ${npv:,.2f} indicates value-destroying investment. "
            interpretation += "Project does not meet required return threshold. "
        else:
            interpretation = "NPV of zero indicates project meets exactly the required return rate. "

        # Add risk considerations
        if len(risk_factors.get("key_risks", [])) > 2:
            interpretation += "Consider high risk factors in decision making."

        return interpretation

    def _calculate_irr(self, numbers: list, question: str) -> Dict[str, Any]:
        """Enhanced IRR calculation with iterative solving and edge case handling."""
        if len(numbers) < 2:
            return {"error": "Need initial investment and cash flows for IRR"}

        initial_investment = abs(numbers[0])
        cash_flows = numbers[1:]

        # Validate cash flows
        validation_result = self._validate_irr_inputs(initial_investment, cash_flows)
        if validation_result.get("error"):
            return validation_result

        # Calculate IRR using Newton-Raphson method
        irr_result = self._calculate_irr_newton_raphson(initial_investment, cash_flows)

        # If Newton-Raphson fails, try bisection method
        if irr_result.get("error") and "convergence" in irr_result.get("error", "").lower():
            irr_result = self._calculate_irr_bisection(initial_investment, cash_flows)

        # If both methods fail, provide fallback approximation
        if irr_result.get("error"):
            irr_result = self._calculate_irr_approximation(initial_investment, cash_flows)
            irr_result["method"] = "approximation_fallback"
            irr_result["warning"] = "Exact IRR calculation failed, using approximation"

        # Add comprehensive analysis
        if not irr_result.get("error"):
            irr_percentage = irr_result["irr"]

            # Calculate additional metrics
            npv_at_irr = self._calculate_npv_at_rate(initial_investment, cash_flows, irr_percentage / 100)
            sensitivity_analysis = self._perform_irr_sensitivity_analysis(initial_investment, cash_flows, irr_percentage)

            irr_result.update({
                "calculation": "Enhanced IRR Analysis",
                "formula": "NPV = 0 = -Initial Investment + Σ(CFt / (1 + IRR)^t)",
                "inputs": {
                    "initial_investment": initial_investment,
                    "cash_flows": cash_flows,
                    "number_of_periods": len(cash_flows)
                },
                "npv_verification": npv_at_irr,
                "sensitivity_analysis": sensitivity_analysis,
                "interpretation": self._interpret_irr_result(irr_percentage, initial_investment, cash_flows),
                "edge_case_analysis": self._analyze_irr_edge_cases(initial_investment, cash_flows)
            })

        return irr_result

    def _validate_irr_inputs(self, initial_investment: float, cash_flows: list) -> Dict[str, Any]:
        """Validate inputs for IRR calculation."""
        if initial_investment <= 0:
            return {"error": "Initial investment must be positive"}

        if not cash_flows:
            return {"error": "Cash flows cannot be empty"}

        if all(cf <= 0 for cf in cash_flows):
            return {"error": "All cash flows are non-positive - IRR undefined"}

        # For IRR calculation, we expect some positive cash flows (normal investment pattern)
        # Only flag as error if there are truly no positive cash flows
        total_cash_flow = sum(cash_flows)
        if total_cash_flow <= 0:
            return {"error": "Total cash flows are negative - project likely unprofitable"}

        return {"valid": True}

    def _calculate_irr_newton_raphson(self, initial_investment: float, cash_flows: list, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, Any]:
        """Calculate IRR using Newton-Raphson method."""
        # Initial guess - use simple approximation
        total_cf = sum(cash_flows)
        years = len(cash_flows)
        initial_guess = ((total_cf / initial_investment) ** (1/years)) - 1

        # Ensure reasonable starting point
        initial_guess = max(min(initial_guess, 10.0), -0.99)  # Clamp between -99% and 1000%

        rate = initial_guess

        for iteration in range(max_iterations):
            # Calculate NPV and its derivative
            npv = -initial_investment
            npv_derivative = 0

            for t, cf in enumerate(cash_flows, 1):
                discount_factor = (1 + rate) ** t
                npv += cf / discount_factor
                npv_derivative -= t * cf / (discount_factor * (1 + rate))

            # Check for convergence
            if abs(npv) < tolerance:
                return {
                    "irr": rate * 100,  # Convert to percentage
                    "iterations": iteration + 1,
                    "method": "newton_raphson",
                    "convergence_achieved": True,
                    "final_npv": npv
                }

            # Newton-Raphson update
            if abs(npv_derivative) < 1e-10:
                return {"error": "Derivative too small - Newton-Raphson cannot converge"}

            rate_new = rate - npv / npv_derivative

            # Prevent rate from going too extreme
            if rate_new < -0.99:
                rate_new = -0.99
            elif rate_new > 10.0:
                rate_new = 10.0

            rate = rate_new

        return {"error": f"Newton-Raphson failed to converge after {max_iterations} iterations"}

    def _calculate_irr_bisection(self, initial_investment: float, cash_flows: list, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, Any]:
        """Calculate IRR using bisection method as fallback."""
        # Find bounds where NPV changes sign
        low_rate = -0.99  # -99%
        high_rate = 10.0  # 1000%

        low_npv = self._calculate_npv_at_rate(initial_investment, cash_flows, low_rate)
        high_npv = self._calculate_npv_at_rate(initial_investment, cash_flows, high_rate)

        # Check if bounds bracket a root
        if low_npv * high_npv > 0:
            # Try to find better bounds
            for test_rate in [-0.5, 0, 0.5, 1.0, 2.0, 5.0]:
                test_npv = self._calculate_npv_at_rate(initial_investment, cash_flows, test_rate)
                if low_npv * test_npv < 0:
                    high_rate = test_rate
                    high_npv = test_npv
                    break
                elif test_npv * high_npv < 0:
                    low_rate = test_rate
                    low_npv = test_npv
                    break
            else:
                return {"error": "Cannot find bounds that bracket IRR root"}

        # Bisection method
        for iteration in range(max_iterations):
            mid_rate = (low_rate + high_rate) / 2
            mid_npv = self._calculate_npv_at_rate(initial_investment, cash_flows, mid_rate)

            if abs(mid_npv) < tolerance:
                return {
                    "irr": mid_rate * 100,  # Convert to percentage
                    "iterations": iteration + 1,
                    "method": "bisection",
                    "convergence_achieved": True,
                    "final_npv": mid_npv
                }

            if low_npv * mid_npv < 0:
                high_rate = mid_rate
                high_npv = mid_npv
            else:
                low_rate = mid_rate
                low_npv = mid_npv

        return {"error": f"Bisection method failed to converge after {max_iterations} iterations"}

    def _calculate_irr_approximation(self, initial_investment: float, cash_flows: list) -> Dict[str, Any]:
        """Fallback approximation method for IRR."""
        total_cash_flow = sum(cash_flows)
        years = len(cash_flows)

        if years == 0:
            return {"error": "No cash flows provided"}

        # Simple approximation: (Total cash flow / Initial investment) ^ (1/years) - 1
        irr_approx = ((total_cash_flow / initial_investment) ** (1/years)) - 1

        return {
            "irr": irr_approx * 100,
            "method": "approximation",
            "note": "This is a simplified approximation - actual IRR may differ"
        }

    def _calculate_npv_at_rate(self, initial_investment: float, cash_flows: list, rate: float) -> float:
        """Calculate NPV at a specific discount rate."""
        npv = -initial_investment
        for t, cf in enumerate(cash_flows, 1):
            npv += cf / ((1 + rate) ** t)
        return npv

    def _perform_irr_sensitivity_analysis(self, initial_investment: float, cash_flows: list, irr_percentage: float) -> Dict[str, Any]:
        """Perform sensitivity analysis around the IRR."""
        base_rate = irr_percentage / 100
        sensitivity_rates = [
            base_rate - 0.05,  # -5%
            base_rate - 0.02,  # -2%
            base_rate - 0.01,  # -1%
            base_rate,         # Base
            base_rate + 0.01,  # +1%
            base_rate + 0.02,  # +2%
            base_rate + 0.05   # +5%
        ]

        sensitivity_results = []
        for rate in sensitivity_rates:
            npv = self._calculate_npv_at_rate(initial_investment, cash_flows, rate)
            sensitivity_results.append({
                "rate_percentage": rate * 100,
                "npv": npv,
                "rate_difference": (rate - base_rate) * 100
            })

        return {
            "sensitivity_table": sensitivity_results,
            "rate_sensitivity": "High" if abs(sensitivity_results[0]["npv"]) > initial_investment * 0.1 else "Moderate"
        }

    def _interpret_irr_result(self, irr_percentage: float, initial_investment: float, cash_flows: list) -> str:
        """Provide interpretation of IRR results."""
        if irr_percentage > 20:
            interpretation = f"High IRR of {irr_percentage:.2f}% indicates excellent returns. "
        elif irr_percentage > 15:
            interpretation = f"Good IRR of {irr_percentage:.2f}% indicates solid returns. "
        elif irr_percentage > 10:
            interpretation = f"Moderate IRR of {irr_percentage:.2f}% indicates acceptable returns. "
        elif irr_percentage > 0:
            interpretation = f"Low IRR of {irr_percentage:.2f}% indicates marginal returns. "
        else:
            interpretation = f"Negative IRR of {irr_percentage:.2f}% indicates the project destroys value. "

        # Add context about typical hurdle rates
        if irr_percentage > 12:
            interpretation += "Exceeds typical corporate hurdle rates."
        elif irr_percentage > 8:
            interpretation += "May meet hurdle rate depending on risk profile."
        else:
            interpretation += "Likely below typical hurdle rate requirements."

        return interpretation

    def _analyze_irr_edge_cases(self, initial_investment: float, cash_flows: list) -> Dict[str, Any]:
        """Analyze potential edge cases in IRR calculation."""
        edge_cases = []

        # Check for sign changes (multiple IRRs possible)
        sign_changes = 0
        for i in range(len(cash_flows)):
            current_cf = cash_flows[i]
            if i == 0:
                prev_sign = -1 if initial_investment > 0 else 1  # Initial investment is negative cash flow
            else:
                prev_sign = 1 if cash_flows[i-1] > 0 else -1

            current_sign = 1 if current_cf > 0 else -1
            if prev_sign != current_sign:
                sign_changes += 1

        if sign_changes > 1:
            edge_cases.append("Multiple sign changes detected - multiple IRRs may exist")

        # Check for very large later cash flows
        if len(cash_flows) > 1:
            early_flows = cash_flows[:2]
            later_flows = cash_flows[2:]
            if later_flows and any(cf > max(early_flows) * 5 for cf in later_flows):
                edge_cases.append("Large terminal cash flows may skew IRR calculation")

        # Check for very small cash flows relative to investment
        total_cf = sum(cash_flows)
        if total_cf < initial_investment * 0.1:
            edge_cases.append("Very small cash flows relative to investment")

        return {
            "potential_issues": edge_cases,
            "sign_changes": sign_changes,
            "multiple_irr_risk": "High" if sign_changes > 1 else "Low"
        }

    def _general_calculation(self, numbers: list, question: str) -> Dict[str, Any]:
        """General calculation based on question context."""
        if not numbers:
            return {"error": "No numbers found in question"}

        question_lower = question.lower()

        # Simple operations
        if len(numbers) >= 2:
            if any(word in question_lower for word in ['add', 'plus', 'sum', 'total']):
                result = sum(numbers)
                operation = "addition"
            elif any(word in question_lower for word in ['subtract', 'minus', 'difference']):
                result = numbers[0] - numbers[1]
                operation = "subtraction"
            elif any(word in question_lower for word in ['multiply', 'times', 'product']):
                result = numbers[0] * numbers[1]
                operation = "multiplication"
            elif any(word in question_lower for word in ['divide', 'divided by']):
                result = numbers[0] / numbers[1] if numbers[1] != 0 else None
                operation = "division"
            elif any(word in question_lower for word in ['percentage', 'percent']):
                result = (numbers[0] / numbers[1]) * 100 if numbers[1] != 0 else None
                operation = "percentage"
            else:
                result = numbers[0]
                operation = "first_number"

            return {
                "calculation": operation,
                "inputs": numbers[:2] if len(numbers) >= 2 else numbers,
                "result": result,
                "numbers_found": numbers
            }

        return {
            "calculation": "number_extraction",
            "result": numbers[0] if numbers else None,
            "numbers_found": numbers
        }

    def _interpret_ratio(self, ratio: float, ratio_type: str) -> str:
        """Provide basic interpretation of ratio results."""
        if ratio_type == "current_ratio":
            if ratio > 2:
                return "High liquidity - may indicate inefficient use of assets"
            elif ratio > 1:
                return "Good liquidity - can cover short-term obligations"
            else:
                return "Low liquidity - may have difficulty paying short-term debts"

        elif ratio_type == "debt_ratio":
            if ratio > 0.5:
                return "High debt level - higher financial risk"
            elif ratio > 0.3:
                return "Moderate debt level"
            else:
                return "Low debt level - conservative financing"

        elif ratio_type == "pe_ratio":
            if ratio > 25:
                return "High P/E - growth expectations or overvalued"
            elif ratio > 15:
                return "Moderate P/E - reasonable valuation"
            else:
                return "Low P/E - undervalued or declining growth"

        else:
            return f"Ratio: {ratio:.2f}"

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            return False

        # Must have either question or operation
        if not input_data.get("question") and not input_data.get("operation"):
            return False

        return True