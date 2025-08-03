![What is it?](what-is-it.jpg)

## 1. Purpose of the Code

Imagine you have a long text document that needs to be processed by AI systems with limited context windows, like those used in RAG (Retrieval-Augmented Generation) pipelines. The **Sentences Chunker** provides a production-ready API service that intelligently breaks down long texts into smaller, optimally-sized segments called "chunks" — while respecting natural language boundaries and maintaining semantic coherence.

Unlike naive text splitters that might cut mid-sentence or mid-word, this tool:

* **Preserves Sentence Integrity**: Uses state-of-the-art neural models (WTPSplit SaT) to detect sentence boundaries across 85+ languages without requiring language-specific configuration
* **Enforces Precise Token Limits**: Ensures chunks fit within model context windows using exact tiktoken-based counting (cl100k_base encoding)
* **Maintains Contextual Continuity**: Configurable sentence overlap between chunks prevents information loss at boundaries
* **Provides Smart Error Handling**: In strict mode, offers intelligent parameter suggestions when constraints cannot be met
* **Optimizes Performance**: GPU acceleration, persistent model caching, and async processing for production workloads

Think of it as a sophisticated text preprocessor that transforms unstructured documents into perfectly-sized, context-aware segments ready for embeddings generation, semantic search, or LLM processing.

## 2. Input(s) it Takes

The API exposes RESTful endpoints that accept HTTP POST requests with:

### Primary Input
* **Text File (`file` parameter)**: Upload a `.txt` or `.md` file containing the document to chunk
  * Encoding: UTF-8 (automatic fallback handling for other encodings)
  * Size: No hard limit, optimized for documents up to several MB

### Configuration Parameters

#### For `/file-chunker/` endpoint:
* **`max_chunk_tokens`** (int, default: 500): Maximum tokens per chunk (must be > 0)
* **`overlap_sentences`** (int, default: 1): Number of sentences to repeat between consecutive chunks (0 = no overlap)
* **`model_name`** (enum, default: "sat-12l-sm"): WTPSplit model variant:
  * Speed-optimized: `sat-1l`, `sat-1l-sm` (single layer)
  * Balanced: `sat-3l`, `sat-3l-sm`, `sat-6l`, `sat-6l-sm`
  * Accuracy-optimized: `sat-9l`, `sat-12l`, `sat-12l-sm`
* **`split_threshold`** (float, default: 0.5): Confidence threshold for sentence boundaries (0.0-1.0)
* **`strict_mode`** (bool, default: false): If true, returns structured error with suggestions when limits cannot be met

#### For `/split-sentences/` endpoint:
* **`model_name`** and **`split_threshold`** (same as above)

## 3. Output(s) it Produces

The API returns structured JSON responses with comprehensive metadata:

### `/file-chunker/` Response Structure

```json
{
  "chunks": [
    {
      "text": "First chunk containing complete sentences...",
      "token_count": 487,
      "id": 1,
      "overflow_details": null  // Present only in non-strict mode if limits exceeded
    }
  ],
  "metadata": {
    "configured_max_chunk_tokens": 500,
    "configured_overlap_sentences": 1,
    "n_input_sentences": 250,         // Pre-chunking statistics
    "avg_tokens_per_input_sentence": 24,
    "max_tokens_in_input_sentence": 89,
    "min_tokens_in_input_sentence": 3,
    "n_chunks": 12,                   // Post-chunking statistics
    "avg_tokens_per_chunk": 485,
    "max_tokens_in_chunk": 500,
    "min_tokens_in_chunk": 156,
    "sat_model_name": "sat-12l-sm",
    "split_threshold": 0.5,
    "source": "document.txt",
    "processing_time": 2.34
  }
}
```

### Strict Mode Error Response (400 status)

```json
{
  "detail": {
    "chunk_process": "failed",
    "single_sentence_too_large": false,
    "overlap_too_large": true,
    "suggested_token_limit": 800,     // Calculated with 30% safety margin
    "suggested_overlap_value": 1      // Maximum viable overlap
  }
}
```

## 4. How it Achieves its Purpose (Core Algorithm)

The chunking pipeline consists of sophisticated stages:

### Stage 1: Intelligent Sentence Segmentation
1. **Text Preprocessing**: Normalizes line breaks (single → space, double → paragraph boundary)
2. **Neural Segmentation**: WTPSplit SaT model analyzes text using transformer architecture
   * Language-agnostic: Works on 85+ languages without configuration
   * Punctuation-agnostic: Handles text without punctuation marks
   * Context-aware: Recognizes abbreviations, URLs, decimal numbers

### Stage 2: Token-Aware Chunking
1. **Precise Token Counting**: Each sentence tokenized with tiktoken (cl100k_base)
2. **Greedy Accumulation**: Sentences grouped into chunks up to max_chunk_tokens
3. **Overlap Management**: Last N sentences from previous chunk prepended to maintain context
4. **Constraint Validation**: 
   * Non-strict mode: Creates oversized chunks with warnings
   * Strict mode: Calculates optimal parameters and returns suggestions

### Stage 3: Performance Optimizations
1. **Model Caching**: 
   * In-memory cache with 1-hour timeout
   * Persistent disk cache (pickle format) in `models/` directory
2. **GPU Acceleration**: Automatic CUDA detection and model placement
3. **Async Processing**: FastAPI with uvloop for high concurrency

## 5. Important Logic Flows & Data Transformations

### Model Loading Flow
```
Request → Check Memory Cache → Found? Use it
                            ↓ Not Found
                    Check Disk Cache → Found? Load to Memory
                                    ↓ Not Found
                            Download from HuggingFace → Save to Disk
```

### Chunking Decision Tree
```
For each sentence:
├─ Would adding it exceed max_tokens?
│  ├─ YES → Finalize current chunk
│  │        → Apply overlap to new chunk
│  │        → Add sentence to new chunk
│  └─ NO → Add to current chunk
│
└─ Single sentence exceeds limit?
   ├─ Strict Mode → Return error with suggestions
   └─ Non-strict → Create oversized chunk with warning
```

### Parameter Suggestion Algorithm (Strict Mode)
1. **For single sentence overflow**:
   * Calculate required tokens + 30% margin
   * Round up to nearest 100 for user-friendly value

2. **For overlap overflow**:
   * Test each possible overlap size (N-1, N-2, ..., 0)
   * Find maximum that fits within current limit
   * Suggest both: increased token limit OR reduced overlap

## 6. Production-Ready Features

### Robustness
* **Comprehensive error handling** with detailed messages
* **Input validation** at multiple levels
* **Encoding fallbacks** for non-UTF8 files
* **Thread-safe** model caching with RLock

### Scalability
* **Docker support** with GPU/CPU profiles
* **Configurable concurrency** limits
* **Memory-efficient** batch processing
* **Automatic GPU memory cleanup**

### Observability
* **Structured logging** with rotating file handler
* **Health check endpoint** with model status
* **Processing time metrics** in responses
* **Detailed metadata** for analysis

### Security
* **Non-root container** execution
* **File type validation** (.txt, .md only)
* **No arbitrary code execution**
* **Resource limits** enforced

This isn't just a text splitter — it's a production-grade text preprocessing service designed for modern NLP pipelines, with the robustness and features required for real-world deployment at scale.
