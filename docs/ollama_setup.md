# Ollama Setup Guide for FinanceQA AI Agent

This guide provides step-by-step instructions for setting up Ollama with models for the FinanceQA AI Agent.

## Prerequisites

- macOS, Linux, or Windows with WSL2
- At least 8GB RAM (16GB+ recommended for larger models)
- At least 10GB free disk space

## Step 1: Install Ollama

### macOS
```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or install via Homebrew
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows (WSL2)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## Step 2: Start Ollama Server

```bash
# Start Ollama server (runs on localhost:11434 by default)
ollama serve
```

The server will start and listen on `http://localhost:11434`. Keep this terminal open.

## Step 3: Download Recommended Models

Open a new terminal and download the recommended models for financial reasoning:

### Primary Models (Choose one based on your hardware)

#### For 8GB+ RAM - Llama 2 7B (Recommended for development)
```bash
ollama pull llama2
```

#### For 16GB+ RAM - Code Llama 7B (Better for financial calculations)
```bash
ollama pull codellama
```

#### For 32GB+ RAM - Llama 2 13B (Higher quality)
```bash
ollama pull llama2:13b
```

### Alternative Models

#### Mistral 7B (Fast and efficient)
```bash
ollama pull mistral
```

#### Phi-2 (Lightweight, good for testing)
```bash
ollama pull phi
```

## Step 4: Verify Installation

### Check Available Models
```bash
ollama list
```

You should see output like:
```
NAME                    ID              SIZE    MODIFIED
llama2:latest          e592a3173145    3.8 GB  2 minutes ago
```

### Test Model Interaction
```bash
# Test with a simple financial question
ollama run llama2 "What is the formula for calculating ROI?"
```

### Verify API Access
```bash
# Test API endpoint
curl http://localhost:11434/api/tags
```

Expected response:
```json
{
  "models": [
    {
      "name": "llama2:latest",
      "modified_at": "2024-01-01T00:00:00Z",
      "size": 3791730596
    }
  ]
}
```

## Step 5: Configure FinanceQA Agent

Create a configuration file `config/model_config.json`:

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
    }
  },
  "default_provider": "ollama"
}
```

## Step 6: Test Integration

Run the OllamaProvider test suite:

```bash
# Run unit tests (mocked)
pytest src/models/ollama_provider.test.py -v

# Run integration tests (requires running Ollama)
pytest src/models/ollama_provider.test.py -v -m integration
```

## Step 7: Performance Optimization

### Adjust Model Parameters

For financial reasoning, consider these optimizations:

```json
{
  "options": {
    "temperature": 0.1,     // Lower for more consistent answers
    "top_p": 0.9,           // Focus on high-probability tokens
    "repeat_penalty": 1.1,   // Reduce repetition
    "num_predict": 1024,     // Allow longer responses for complex calculations
    "num_ctx": 4096         // Larger context for document analysis
  }
}
```

### Memory Management

For limited RAM systems:
```bash
# Use smaller models
ollama pull phi        # ~2.7GB
ollama pull mistral:7b-instruct-v0.1-q4_0  # ~4GB quantized
```

## Troubleshooting

### Common Issues

#### 1. "Connection refused" error
- Ensure Ollama server is running: `ollama serve`
- Check if port 11434 is available: `netstat -an | grep 11434`

#### 2. Model not found
```bash
# List available models
ollama list

# Pull the required model
ollama pull llama2
```

#### 3. Out of memory errors
- Use smaller models (phi, mistral:7b)
- Close other memory-intensive applications
- Consider quantized models (q4_0, q5_0 versions)

#### 4. Slow responses
- Use SSD storage for model files
- Ensure sufficient RAM
- Consider GPU acceleration if available

### Performance Monitoring

Monitor Ollama performance:
```bash
# Check system resources
top -p $(pgrep ollama)

# Monitor API responses
curl -w "%{time_total}s\n" http://localhost:11434/api/tags
```

## Model Recommendations by Use Case

### Development & Testing
- **phi**: Fastest, lowest resource usage
- **mistral**: Good balance of speed and quality

### Production Financial Reasoning
- **llama2**: Reliable, well-tested
- **codellama**: Better at mathematical reasoning
- **llama2:13b**: Higher quality for complex analysis

### Document Analysis
- **llama2:13b**: Better context understanding
- **mistral**: Fast processing of large documents

## Next Steps

1. Test the integration with your FinanceQA agent
2. Benchmark performance on sample financial questions
3. Fine-tune parameters based on your specific use cases
4. Consider setting up model caching for production use

## Useful Commands Reference

```bash
# Server management
ollama serve                    # Start server
ollama ps                       # Show running models

# Model management
ollama pull <model>             # Download model
ollama list                     # List installed models
ollama rm <model>               # Remove model

# Usage
ollama run <model> <prompt>     # Interactive chat
ollama run <model> < file.txt   # Process file

# API testing
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "prompt": "What is ROI?", "stream": false}'
```