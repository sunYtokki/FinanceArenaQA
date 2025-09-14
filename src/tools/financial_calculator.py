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
        """Simple NPV calculation."""
        if len(numbers) < 2:
            return {"error": "Need initial investment and cash flows for NPV"}

        # Simplified NPV assuming 10% discount rate
        discount_rate = 0.10
        initial_investment = numbers[0]
        cash_flows = numbers[1:]

        npv = -initial_investment
        for i, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** (i + 1))

        return {
            "calculation": "NPV",
            "formula": "Sum of discounted cash flows - initial investment",
            "inputs": {
                "initial_investment": initial_investment,
                "cash_flows": cash_flows,
                "discount_rate": discount_rate
            },
            "result": npv,
            "interpretation": "Positive NPV indicates profitable investment"
        }

    def _calculate_irr(self, numbers: list, question: str) -> Dict[str, Any]:
        """Simple IRR approximation."""
        if len(numbers) < 2:
            return {"error": "Need initial investment and cash flows for IRR"}

        # Simplified IRR approximation
        initial_investment = abs(numbers[0])
        cash_flows = numbers[1:]
        total_cash_flow = sum(cash_flows)

        if len(cash_flows) > 0:
            # Simple approximation: (Total cash flow / Initial investment) ^ (1/years) - 1
            years = len(cash_flows)
            irr_approx = ((total_cash_flow / initial_investment) ** (1/years)) - 1

            return {
                "calculation": "IRR (approximation)",
                "formula": "Approximate IRR calculation",
                "inputs": {
                    "initial_investment": initial_investment,
                    "cash_flows": cash_flows,
                    "years": years
                },
                "result": irr_approx * 100,  # As percentage
                "unit": "percentage",
                "note": "This is a simplified approximation"
            }

        return {"error": "Invalid cash flows for IRR calculation"}

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