![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green)
![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-blue)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.12-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

![Sentences Chunker](logo.png)

# Sentences Chunker

The Sentences Chunker is a cutting-edge tool that revolutionizes text segmentation for modern NLP applications by intelligently splitting documents into optimally-sized chunks while preserving sentence boundaries and semantic integrity.
This innovative solution leverages state-of-the-art [WTPSplit (Where's the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation)](https://github.com/segment-any-text/wtpsplit) technology to deliver unparalleled accuracy across 85+ languages without requiring language-specific models or punctuation.
Traditional text chunkers often break sentences arbitrarily at token limits, destroying context and meaning. This leads to degraded performance in downstream tasks like embeddings generation, retrieval-augmented generation (RAG), and language model processing.
The Sentences Chunker overcomes these challenges by combining advanced sentence boundary detection with intelligent token-aware chunking. It ensures chunks respect natural language boundaries while adhering to strict token limits, with configurable sentence overlap for maintaining context across chunk boundaries.
Whether you're building RAG pipelines, preparing training data, or optimizing text for LLM consumption, the Sentences Chunker provides a robust, production-ready solution for intelligent text segmentation.

## Key Features
-   **Advanced Sentence Segmentation**: Powered by WTPSplit's neural models for state-of-the-art sentence boundary detection across 85+ languages.
-   **Flexible Model Selection**: Choose from multiple SaT (Segment any Text) models optimized for different speed/accuracy trade-offs (1-layer to 12-layer variants).
-   **Precise Token Control**: Enforce strict token limits per chunk while preserving sentence integrity.
-   **Configurable Sentence Overlap**: Maintain contextual continuity between chunks with customizable overlap settings.
-   **Strict Mode with Smart Suggestions**: Get intelligent parameter recommendations when chunking constraints cannot be met.
-   **GPU Acceleration**: CUDA-enabled for fast processing with automatic GPU/CPU detection.
-   **Persistent Model Caching**: Models are saved to disk after first download for instant subsequent loads.
-   **Comprehensive Metadata**: Detailed statistics on chunking results including token distribution and processing metrics.
-   **Universal REST API with FastAPI**: Modern, high-performance API interface with automatic documentation, data validation, and seamless integration capabilities for any system or language.
-   **Docker Integration**: Easy deployment with GPU/CPU profiles and automatic hardware detection.

## Table of Contents

- [How the Text Chunking Algorithm Works](#how-the-text-chunking-algorithm-works)
  - [The Pipeline](#the-pipeline)
  - [Intelligent Overlap Management](#intelligent-overlap-management)
  - [Strict Mode and Parameter Optimization](#strict-mode-and-parameter-optimization)
  - [Comparison with Traditional Chunking](#comparison-with-traditional-chunking)
- [Advantages of the Solution](#advantages-of-the-solution)
  - [Superior Sentence Segmentation](#superior-sentence-segmentation)
  - [Optimal for RAG and LLM Applications](#optimal-for-rag-and-llm-applications)
  - [Performance and Scalability](#performance-and-scalability)
- [Installation and Deployment](#installation-and-deployment)
  - [Prerequisites](#prerequisites)
  - [Getting the Code](#getting-the-code)
  - [Local Installation with Uvicorn](#local-installation-with-uvicorn)
  - [Docker Deployment (Recommended)](#docker-deployment-recommended)
- [Using the API](#using-the-api)
  - [API Endpoints](#api-endpoints)
  - [Example API Call](#example-api-call)
  - [Response Format](#response-format)
- [Contributing](#contributing)

## How the Text Chunking Algorithm Works

### The Pipeline

The Sentences Chunker implements a sophisticated multi-stage pipeline that combines neural sentence segmentation with intelligent chunking:

1. The application exposes a REST API where users upload text documents with parameters for token limits, overlap settings, and model selection.
2. Text preprocessing handles single line breaks while preserving paragraph boundaries for optimal sentence detection.
3. The WTPSplit SaT model performs neural sentence segmentation, handling complex cases like abbreviations, URLs, and multilingual text.
4. Each sentence is tokenized using the cl100k_base encoding (compatible with modern LLMs) to calculate precise token counts.
5. An intelligent chunking algorithm groups sentences while respecting the maximum token limit.
6. Optional sentence overlap is applied between chunks to maintain context continuity.
7. In strict mode, the system validates all constraints and provides smart parameter suggestions if limits cannot be met.
8. The API returns structured JSON with chunks, token counts, and comprehensive metadata.

### Intelligent Overlap Management

The overlap feature ensures contextual continuity between chunks, critical for RAG applications:

```python
# Take last N sentences from previous chunk as overlap
# previous_chunk_sentences contains sentences from the last finalized chunk
overlap_start_index = max(0, len(previous_chunk_sentences) - configured_overlap_sentences)
overlap_sentences = previous_chunk_sentences[overlap_start_index:]
```

When overlap would exceed token limits, the system:
- In normal mode: Creates the chunk with a warning in overflow_details
- In strict mode: Calculates optimal parameters and suggests adjustments with safety margins

### Strict Mode and Parameter Optimization

Strict mode ensures absolute compliance with constraints and provides intelligent suggestions:

```python
# Calculate suggested max_chunk_tokens with 30% safety margin
tokens_with_margin = base_required_tokens * 1.3
suggested_max_t = round_up_to_nearest_100(tokens_with_margin)

# Provide actionable suggestions
suggestions = {
    "suggested_max_chunk_tokens": suggested_max_t,
    "suggested_overlap_sentences": optimal_overlap,
    "message": detailed_explanation
}
```

### Comparison with Traditional Chunking

| Feature | Traditional Chunking | Sentences Chunker |
|---------|---------------------|-------------------|
| Sentence Detection | Basic regex or newline splits | Neural model with 85+ language support |
| Boundary Preservation | Often breaks mid-sentence | Always preserves sentence boundaries |
| Token Counting | Approximate or character-based | Precise tiktoken-based counting |
| Overlap Handling | Fixed character/token overlap | Intelligent sentence-based overlap |
| Error Handling | Basic validation | Smart parameter suggestions |
| Model Persistence | Downloads every run | Caches models to disk |
| GPU Support | Rarely implemented | Automatic GPU/CPU detection |

## Advantages of the Solution

### Superior Sentence Segmentation

WTPSplit's neural models provide:
- **Language Agnostic**: Works across 85+ languages without configuration
- **Punctuation Agnostic**: Handles text without punctuation marks
- **Context Aware**: Understands abbreviations, URLs, and special cases
- **Configurable Confidence**: Adjustable split threshold for different use cases

### Optimal for RAG and LLM Applications

Chunks generated are ideal for modern NLP pipelines:
- **Semantic Integrity**: Complete sentences preserve meaning and context
- **Token Precision**: Exact token counts ensure compatibility with model limits
- **Context Windows**: Overlap maintains continuity for retrieval tasks
- **Metadata Rich**: Detailed statistics for pipeline optimization

### Performance and Scalability

Production-ready features include:
- **GPU Acceleration**: Automatic CUDA detection and optimization
- **Model Caching**: Persistent storage eliminates redundant downloads
- **Async Processing**: FastAPI with uvloop for high concurrency
- **Memory Management**: Efficient GPU memory handling with automatic cleanup
- **Docker Profiles**: Separate CPU/GPU deployments with health checks

## Installation and Deployment

### Prerequisites

- Docker and Docker Compose (for Docker deployment)
- NVIDIA GPU with CUDA support (recommended for performance)
- NVIDIA Container Toolkit (for GPU passthrough in Docker)
- Python 3.10-3.12 (for local installation)

### Getting the Code

Before proceeding with any installation method, clone the repository:
```bash
git clone https://github.com/yourusername/sentences-chunker.git
cd sentences-chunker
```

### Local Installation with Uvicorn

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```
   
   **For Windows users:**
   
   * Using Command Prompt:
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   * Using PowerShell:
   ```powershell
   # If you encounter execution policy restrictions, run this once per session:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   
   # Then activate the virtual environment:
   venv\Scripts\Activate.ps1
   ```
   > **Note:** PowerShell's default security settings may prevent script execution. The above command temporarily allows scripts for the current session only, which is safer than changing system-wide settings.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: For GPU support, ensure you install the correct PyTorch version:
   ```bash
   pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.1.1+cu121
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn sentences_chunker:app --reload
   ```

4. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.

### Docker Deployment (Recommended)

1. Create required directories for persistent storage:
   ```bash
   # Linux/macOS
   mkdir -p models logs
   
   # Windows CMD
   mkdir models
   mkdir logs
   
   # Windows PowerShell
   New-Item -ItemType Directory -Path models -Force
   New-Item -ItemType Directory -Path logs -Force
   # Or with PowerShell alias
   mkdir -Force models, logs
   ```

2. Deploy with Docker Compose:

   **CPU-only deployment** (default, works on all machines):
   ```bash
   cd docker
   docker compose --profile cpu up -d
   ```

   **GPU-accelerated deployment** (requires NVIDIA GPU and drivers):
   ```bash
   cd docker
   docker compose --profile gpu up -d
   ```

   **Stopping the service**:
   
   > **Important**: Always match the `--profile` flag between your `up` and `down` commands:
   ```bash
   # To stop CPU deployment
   docker compose --profile cpu down
   
   # To stop GPU deployment
   docker compose --profile gpu down
   ```
   > This ensures Docker Compose correctly identifies and manages the specific set of containers you intended to control.

3. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.

## Using the API

### API Endpoints

- **POST `/file-chunker/`**  
  Chunks a text document into segments based on specified token limits with optional sentence overlap.
  
  **Parameters:**
  - `file`: The text file to be chunked (supports .txt and .md formats)
  - `model_name`: WTPSplit SaT model to use (default: sat_1l_sm)
  - `split_threshold`: Confidence threshold for sentence boundaries (0.0-1.0, default: 0.5)
  - `max_chunk_tokens`: Maximum tokens per chunk (integer, default: 500)
  - `overlap_sentences`: Number of sentences to overlap between chunks (0-3, where 0 disables overlap, default: 1)
  - `strict_mode`: If true, enforces all constraints strictly (boolean, default: false)
  
  **Response:**
  Returns a JSON object containing:
  - `chunks`: Array of text segments with token counts, IDs, and overflow details
  - `metadata`: Comprehensive processing statistics

- **POST `/split-sentences/`**  
  Splits text into individual sentences without chunking.
  
  **Parameters:**
  - `file`: The text file to split (supports .txt and .md formats)
  - `model_name`: WTPSplit SaT model to use (default: sat-12l-sm)
  - `split_threshold`: Confidence threshold for boundaries (0.0-1.0, default: 0.5)
  
  **Response:**
  Returns sentences as individual chunks with token counts and metadata.

- **GET `/`**  
  Health check endpoint that returns service status, GPU availability, saved models, and API version.

### Example API Call using cURL

```bash
# Basic file chunking with defaults
curl -X POST "http://localhost:8000/file-chunker/" \
  -F "file=@document.txt" 

# Advanced chunking with all parameters
curl -X POST "http://localhost:8000/file-chunker/?\
max_chunk_tokens=1024&\
overlap_sentences=2&\
model_name=sat-6l&\
split_threshold=0.6&\
strict_mode=true" \
  -F "file=@document.txt" \
  -H "accept: application/json"

# Split into sentences only
curl -X POST "http://localhost:8000/split-sentences/?model_name=sat-3l" \
  -F "file=@document.txt"

# Health check with model information
curl http://localhost:8000/
```

### Example API Call using Python

```python
import requests
import json

# API configuration
api_url = 'http://localhost:8000/file-chunker/'
file_path = 'document.txt'  # Your input text file

# Chunking parameters
params = {
    'max_chunk_tokens': 512,
    'overlap_sentences': 2,
    'model_name': 'sat-12l-sm',  # Best accuracy
    'split_threshold': 0.5,
    'strict_mode': True  # Enforce constraints
}

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/plain')}
        response = requests.post(api_url, files=files, params=params)
        response.raise_for_status()

        result = response.json()
        
        # Handle successful response
        print(f"Successfully chunked into {result['metadata']['n_chunks']} chunks")
        print(f"Average tokens per chunk: {result['metadata']['avg_tokens_per_chunk']}")
        print(f"Processing time: {result['metadata']['processing_time']:.2f}s")
        
        # Save results
        with open('chunks_output.json', 'w', encoding='utf-8') as out:
            json.dump(result, out, indent=2, ensure_ascii=False)

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        # Handle strict mode violations
        error_detail = e.response.json()['detail']
        if isinstance(error_detail, dict) and error_detail.get('chunk_process') == 'failed':
            print("Chunking failed due to constraints. Suggestions:")
            if error_detail.get('suggested_token_limit'):
                print(f"  - Try max_chunk_tokens={error_detail['suggested_token_limit']}")
            if error_detail.get('suggested_overlap_value') is not None:
                print(f"  - Try overlap_sentences={error_detail['suggested_overlap_value']}")
    else:
        print(f"API error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

### Response Format

A successful chunking operation returns a `FileChunkingResult` object:

```json
{
  "chunks": [
    {
      "text": "This is the first chunk containing complete sentences...",
      "token_count": 487,
      "id": 1,
      "overflow_details": null
    },
    {
      "text": "The second chunk starts with overlap sentences from the previous chunk...",
      "token_count": 502,
      "id": 2,
      "overflow_details": null
    }
  ],
  "metadata": {
    "file": "document.txt",
    "configured_max_chunk_tokens": 512,
    "configured_overlap_sentences": 2,
    "n_input_sentences": 150,
    "avg_tokens_per_input_sentence": 24,
    "max_tokens_in_input_sentence": 89,
    "min_tokens_in_input_sentence": 5,
    "n_chunks": 8,
    "avg_tokens_per_chunk": 495,
    "max_tokens_in_chunk": 512,
    "min_tokens_in_chunk": 234,
    "sat_model_name": "sat-12l-sm",
    "split_threshold": 0.5,
    "processing_time": 2.34
  }
}
```

For strict mode violations, the API returns a 400 status with suggestions:

```json
{
  "detail": {
    "chunk_process": "failed",
    "single_sentence_too_large": false,
    "overlap_too_large": true,
    "suggested_token_limit": 800,
    "suggested_overlap_value": 1
  }
}
```

## Contributing

The Sentences Chunker is an open-source project that thrives on community contributions. Your involvement is not just welcome, it's fundamental to the project's growth, innovation, and long-term success.

Whether you're fixing bugs, improving documentation, adding new features, or sharing ideas, every contribution helps build a better tool for everyone. We believe in the power of collaborative development and welcome participants of all skill levels.

If you're interested in contributing:

1. Fork the repository
2. Create a development environment with all dependencies
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a pull request

Happy Sentence Chunking!

---
