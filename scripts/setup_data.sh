#!/bin/bash

# =====================================================================
# FinanceQA Data Setup Script
# =====================================================================
#
# One-time setup script to prepare all data for FinanceQA benchmarking.
# After running this script, you can run the benchmark runner multiple times.
#
# This script performs:
# 1. Downloads FinanceQA benchmark dataset
# 2. Downloads Kaggle financial Q&A dataset
# 3. Processes Q&A data for RAG ingestion
# 4. Ingests data into ChromaDB
#
# Usage:
#   ./scripts/setup_data.sh [options]
#
# Options:
#   --force-clean     Remove existing data and start fresh
#   --skip-download   Skip download if data already exists
#   --help           Show this help message
#
# After completion, use these commands to run benchmarks:
#   python3 src/evaluation/run_agent.py                    # Full benchmark
#   python3 src/evaluation/run_agent.py --num-samples 10   # Quick test
#   python3 src/evaluation/run_llm_evaluation.py --input-file results/your_results.json
#
# =====================================================================

set -euo pipefail

# Configuration
FORCE_CLEAN=false
SKIP_DOWNLOAD=false

# Directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
CHROMA_DB_DIR="$PROJECT_ROOT/chroma_db"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    local level="$1"
    shift
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $*" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $*" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $*" ;;
        "STEP")  echo -e "${BLUE}[STEP]${NC} $*" ;;
    esac
}

show_help() {
    cat << EOF
FinanceQA Data Setup Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --force-clean    Remove existing data and start fresh
    --skip-download  Skip download if data already exists
    --help          Show this help message

DESCRIPTION:
    One-time setup script to prepare all data for FinanceQA benchmarking.
    Sets up FinanceQA benchmark data and RAG knowledge base.

AFTER SETUP:
    Run benchmarks with:
    python3 src/evaluation/run_agent.py
    python3 src/evaluation/run_agent.py --question-type assumption --num-samples 50

    Evaluate results with:
    python3 src/evaluation/run_llm_evaluation.py --input-file results/your_results.json

EOF
}

check_environment() {
    log "INFO" "Checking environment..."

    if ! command -v python3 &> /dev/null; then
        log "ERROR" "Python 3 is required but not found"
        exit 1
    fi

    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log "ERROR" "Must be run from FinanceQA project root"
        exit 1
    fi

    log "INFO" "Environment check passed"
}

clean_existing_data() {
    if [[ "$FORCE_CLEAN" == true ]]; then
        log "WARN" "Cleaning existing data..."
        rm -rf "$DATA_DIR/datasets"
        rm -rf "$CHROMA_DB_DIR"
        log "INFO" "Data cleanup completed"
    fi
}

create_directories() {
    log "INFO" "Creating directories..."
    mkdir -p "$DATA_DIR/datasets/financeqa"
    mkdir -p "$DATA_DIR/datasets/financial_qa"
}

download_datasets() {
    log "STEP" "=== DOWNLOADING DATASETS ==="

    if [[ "$SKIP_DOWNLOAD" == true ]] && [[ -d "$DATA_DIR/datasets/financeqa" ]]; then
        log "INFO" "Skipping download - data already exists"
        return 0
    fi

    log "INFO" "Downloading FinanceQA benchmark and Kaggle Q&A datasets..."

    cd "$PROJECT_ROOT"

    if python3 scripts/download_financeqa.py; then
        log "INFO" "✅ Datasets downloaded successfully"

        # Check what was downloaded
        local financeqa_files=$(find "$DATA_DIR/datasets/financeqa" -name "*.jsonl" 2>/dev/null | wc -l)
        local kaggle_files=$(find "$DATA_DIR/datasets/financial_qa" -name "*.csv" 2>/dev/null | wc -l)

        log "INFO" "Files: FinanceQA ($financeqa_files splits), Kaggle ($kaggle_files files)"
    else
        log "ERROR" "❌ Dataset download failed"
        return 1
    fi
}

process_qa_data() {
    log "STEP" "=== PROCESSING Q&A DATA ==="

    local kaggle_csv="$DATA_DIR/datasets/financial_qa/Financial-QA-10k.csv"
    local processed_json="$DATA_DIR/datasets/financial_qa/processed_qa_pairs.json"

    if [[ ! -f "$kaggle_csv" ]]; then
        log "WARN" "Kaggle Q&A CSV not found, skipping processing"
        return 0
    fi

    if [[ -f "$processed_json" ]] && [[ "$FORCE_CLEAN" != true ]]; then
        log "INFO" "Processed Q&A data already exists"
        return 0
    fi

    log "INFO" "Processing Kaggle Q&A dataset for RAG..."

    cd "$PROJECT_ROOT"

    if python3 -c "
import sys
sys.path.append('.')
from src.rag.data_processor import process_kaggle_dataset

success = process_kaggle_dataset(
    input_file='$kaggle_csv',
    output_file='$processed_json'
)
exit(0 if success else 1)
"; then
        log "INFO" "✅ Q&A data processed successfully"

        if [[ -f "$processed_json" ]]; then
            local qa_count=$(python3 -c "
import json
with open('$processed_json', 'r') as f:
    data = json.load(f)
    print(data['dataset_info']['total_pairs'])
")
            log "INFO" "Processed $qa_count Q&A pairs"
        fi
    else
        log "ERROR" "❌ Q&A data processing failed"
        return 1
    fi
}

ingest_to_chromadb() {
    log "STEP" "=== INGESTING TO CHROMADB ==="

    local processed_json="$DATA_DIR/datasets/financial_qa/processed_qa_pairs.json"

    if [[ ! -f "$processed_json" ]]; then
        log "WARN" "No processed Q&A data found, skipping ingestion"
        return 0
    fi

    log "INFO" "Ingesting data into ChromaDB for RAG..."

    cd "$PROJECT_ROOT"

    if python3 src/rag/ingest_data.py; then
        log "INFO" "✅ Data ingested successfully"

        # Check collection size
        local collection_size=$(python3 -c "
import sys
sys.path.append('.')
try:
    from src.rag.financial_rag import FinancialRAG
    rag = FinancialRAG()
    if rag.is_available():
        stats = rag.get_collection_stats()
        print(stats.get('total_documents', 0))
    else:
        print('0')
except:
    print('0')
")
        log "INFO" "ChromaDB collection: $collection_size documents"
    else
        log "ERROR" "❌ Data ingestion failed"
        return 1
    fi
}

verify_setup() {
    log "STEP" "=== VERIFYING SETUP ==="

    local errors=0

    # Check FinanceQA benchmark data
    if [[ ! -f "$DATA_DIR/datasets/financeqa/test.jsonl" ]]; then
        log "ERROR" "FinanceQA test data missing"
        ((errors++))
    else
        local test_count=$(wc -l < "$DATA_DIR/datasets/financeqa/test.jsonl")
        log "INFO" "FinanceQA test set: $test_count questions"
    fi

    # Check ChromaDB
    local collection_size=$(python3 -c "
import sys
sys.path.append('.')
try:
    from src.rag.financial_rag import FinancialRAG
    rag = FinancialRAG()
    if rag.is_available():
        stats = rag.get_collection_stats()
        print(stats.get('total_documents', 0))
    else:
        print('0')
except:
    print('0')
")

    if [[ "$collection_size" -eq 0 ]]; then
        log "WARN" "ChromaDB collection is empty (RAG will be disabled)"
    else
        log "INFO" "ChromaDB ready: $collection_size documents"
    fi

    if [[ $errors -eq 0 ]]; then
        log "INFO" "✅ Setup verification passed"
        return 0
    else
        log "ERROR" "❌ Setup verification failed with $errors errors"
        return 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-clean)
            FORCE_CLEAN=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    local start_time=$(date +%s)

    log "INFO" "🚀 Starting FinanceQA data setup..."

    check_environment
    clean_existing_data
    create_directories
    download_datasets
    process_qa_data
    ingest_to_chromadb
    verify_setup

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log "INFO" "✅ Setup completed in ${duration}s"
}

# Run with error handling
trap 'log "ERROR" "Setup failed at line $LINENO"; exit 1' ERR
main