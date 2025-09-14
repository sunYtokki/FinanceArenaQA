"""Setup configuration for FinanceQA AI Agent."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="financeqa-ai-agent",
    version="0.1.0",
    description="AI agent for improving FinanceQA benchmark performance through multi-step reasoning",
    author="Sun Kim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "financeqa-agent=agent.cli:main",
        ],
    },
)