![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

# Sentences Chunker

The Sentences Chunker is a cutting-edge tool designed to intelligently segment text documents into optimally-sized chunks. It prioritizes sentence boundaries to preserve natural language structure and context, while adhering to specified token limits. This makes it an ideal tool for preparing text data for downstream Natural Language Processing (NLP) tasks, especially for ingestion by Large Language Models (LLMs) and for creating effective knowledge bases for Retrieval Augmented Generation (RAG) systems.

## Key Features

-   **Flexible Sentence Splitting**: Offers two distinct methods for initial sentence segmentation:
    -   **Newline-based Splitting**: A straightforward approach treating each line (separated by `\n`) as a distinct sentence. Ideal for texts where line breaks intentionally demarcate sentences.
    -   **Regex-based Splitting**: Utilizes a sophisticated regular expression to identify sentence boundaries, handling common edge cases like abbreviations, titles, and decimal numbers for more nuanced segmentation.
-   **Token-Constrained Chunking**: Allows precise control over the maximum number of tokens permitted in each chunk. Ensures chunks are compatible with model context window limitations.
-   **Adaptive Token Limiting**: Automatically determines the minimum tokens required to successfully chunk the document with a specified sentence overlap. This feature produces the densest possible chunks while respecting the desired overlap, optimizing token utilization.
-   **Configurable Sentence Overlap**: Enables defining a specific number of sentences from the end of one chunk to be included at the beginning of the next. This helps maintain contextual continuity across chunks.
-   **Comprehensive Processing Metadata**: Returns detailed metadata with each chunking result, including:
    -   Original filename
    -   Total number of chunks generated
    -   Average, minimum, and maximum token counts across chunks
    -   The max chunk tokens setting used (either user-specified or adaptively determined)
    -   The sentence overlap setting used
    -   Total processing time
-   **Robust and Easy-to-Integrate API**: Built with FastAPI, offering a clean, well-documented RESTful interface for seamless integration into various workflows. Includes structured logging and robust error handling.

## Installation and Deployment

### Prerequisites
- Docker and Docker Compose (for Docker deployment)
- Python 3.10 or higher

### Getting the Code

Before proceeding with any installation method, clone the repository:
```bash
git clone https://github.com/smart-models/Sentences-Chunker.git
cd Sentences-Chunker
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
   .venv\Scripts\activate.bat
   ```
   
   * Using PowerShell:
   ```powershell
   # If you encounter execution policy restrictions, run this once per session:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   
   # Then activate the virtual environment:
   .venv\Scripts\Activate.ps1
   ```
   > **Note:** PowerShell's default security settings may prevent script execution. The above command temporarily allows scripts for the current session only, which is safer than changing system-wide settings.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
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
   mkdir -p logs
   
   # Windows CMD
   mkdir logs
   
   # Windows PowerShell
   New-Item -ItemType Directory -Path models -Force
   New-Item -ItemType Directory -Path logs -Force
   # Or with PowerShell alias
   mkdir -Force models, logs
   ```

2. Deploy with Docker Compose:

   ```bash
   cd docker
   docker compose up -d
   ```

   **Stopping the service**:
   
   ```bash
   docker compose down
   ```

3. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.


## Using the API

### API Endpoints

#### **`GET /`**
Health check endpoint that returns the service status and API version.

**Response:**
Returns a JSON object containing:
*   `status`: Current health status of the service (e.g., "healthy").
*   `version`: The current version of the API.

---

#### **`POST /file-chunker/`**
Chunks a text document into segments based on specified token limits and sentence overlap. Users can choose between newline-based or regex-based sentence splitting.

**Parameters:**

*   `file`: (File, required) The text file to be chunked. Supported formats: `.txt`, `.md`.
*   `max_chunk_tokens`: (Query, integer, optional, default: 800) Maximum number of tokens per chunk. Must be greater than 0.
*   `overlap`: (Query, integer, optional, default: 0) Number of sentences to overlap between consecutive chunks. Must be non-negative.
*   `use_newline_splitting`: (Query, boolean, optional, default: `False`) If `True`, splits text at newline characters. If `False`, uses regex pattern matching for sentence splitting.

**Response:**
Returns a `ChunkingResult` JSON object containing:
*   `chunks`: An array of `Chunk` objects, each with `text`, `token_count`, and `id`.
*   `metadata`: A `ChunkingMetadata` object with statistics about the chunking process, including original filename, chunk count, token statistics for chunks, the `max_chunk_tokens` setting used, overlap setting, and processing time.

---

#### **`POST /adaptive-file-chunking/`**
Chunks a text document using the *minimum required* token limit necessary to successfully segment the text with a specified sentence overlap. This approach aims for the densest possible chunks. Users can choose between newline-based or regex-based sentence splitting.

**Parameters:**

*   `file`: (File, required) The text file to be chunked. Supported formats: `.txt`, `.md`.
*   `overlap`: (Query, integer, required) Number of sentences to overlap between consecutive chunks. Must be between 1 and 3 (inclusive).
*   `use_newline_splitting`: (Query, boolean, optional, default: `False`) If `True`, splits text at newline characters. If `False`, uses regex pattern matching for sentence splitting.

**Response:**
Returns a `ChunkingResult` JSON object containing:
*   `chunks`: An array of `Chunk` objects, each with `text`, `token_count`, and `id`.
*   `metadata`: A `ChunkingMetadata` object with statistics about the chunking process. The `max_chunk_tokens` field in the metadata will reflect the adaptively determined minimum token limit.

---

### Example API Call using cURL

**Health check endpoint**
```bash 
curl http://localhost:8000/
```

**Basic usage for /file-chunker/ (uses defaults for overlap and use_newline_splitting)**
```bash
curl -X POST "http://localhost:8000/file-chunker/?max_chunk_tokens=500" \
  -F "file=@document.txt" \
  -H "accept: application/json"
```

**With all parameters specified for /file-chunker/**
```bash
curl -X POST "http://localhost:8000/file-chunker/?max_chunk_tokens=500&overlap=1&use_newline_splitting=true" \
  -F "file=@document.txt" \
  -H "accept: application/json"
```

**Basic usage for /adaptive-file-chunking/ (uses default for use_newline_splitting)**
```bash
curl -X POST "http://localhost:8000/adaptive-file-chunking/?overlap=2" \
  -F "file=@document.txt" \
  -H "accept: application/json"
```

**With all parameters specified for /adaptive-file-chunking/**
```bash
curl -X POST "http://localhost:8000/adaptive-file-chunking/?overlap=2&use_newline_splitting=false" \
  -F "file=@document.txt" \
  -H "accept: application/json"
```

### Example API Call using Python

```python
import requests
import json

# Replace with your actual API base URL if hosted elsewhere
api_base_url = 'http://localhost:8000'
file_path = 'document.txt' # Your input text file

# --- Example for /file-chunker/ ---
print("--- Testing /file-chunker/ endpoint ---")
file_chunker_url = f"{api_base_url}/file-chunker/"
params_file_chunker = {
    'max_chunk_tokens': 500,
    'overlap': 1,
    'use_newline_splitting': False
}

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/plain')}
        response = requests.post(file_chunker_url, files=files, params=params_file_chunker)
        response.raise_for_status() # Raise an exception for bad status codes

        result = response.json()
        print(f"Successfully chunked document using /file-chunker/ into {result['metadata']['n_chunks']} chunks.")
        # print("Metadata:", result['metadata'])
        # print("First chunk:", result['chunks'][0] if result['chunks'] else "No chunks")

        output_file_fc = 'response_file_chunker.json'
        with open(output_file_fc, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)
        print(f"Response from /file-chunker/ saved to {output_file_fc}\n")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except requests.exceptions.RequestException as e:
    print(f"API Request to /file-chunker/ failed: {e}")
    if e.response is not None:
        print("Error details:", e.response.text)
except Exception as e:
    print(f"An unexpected error occurred with /file-chunker/: {e}")

# --- Example for /adaptive-file-chunking/ ---
print("--- Testing /adaptive-file-chunking/ endpoint ---")
adaptive_chunking_url = f"{api_base_url}/adaptive-file-chunking/"
params_adaptive = {
    'overlap': 2,
    'use_newline_splitting': False
}

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/plain')}
        response = requests.post(adaptive_chunking_url, files=files, params=params_adaptive)
        response.raise_for_status() # Raise an exception for bad status codes

        result = response.json()
        print(f"Successfully chunked document using /adaptive-file-chunking/ into {result['metadata']['n_chunks']} chunks.")
        print(f"Adaptive method determined max_chunk_tokens: {result['metadata']['max_chunk_tokens']}")
        # print("Metadata:", result['metadata'])
        # print("First chunk:", result['chunks'][0] if result['chunks'] else "No chunks")

        output_file_adaptive = 'response_adaptive_chunking.json'
        with open(output_file_adaptive, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)
        print(f"Response from /adaptive-file-chunking/ saved to {output_file_adaptive}\n")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except requests.exceptions.RequestException as e:
    print(f"API Request to /adaptive-file-chunking/ failed: {e}")
    if e.response is not None:
        print("Error details:", e.response.text)
except Exception as e:
    print(f"An unexpected error occurred with /adaptive-file-chunking/: {e}")

# --- Example for Health Check / ---
print("--- Testing / (Health Check) endpoint ---")
health_check_url = f"{api_base_url}/"
try:
    response = requests.get(health_check_url)
    response.raise_for_status()
    health_status = response.json()
    print("Health Check Status:", health_status)
except requests.exceptions.RequestException as e:
    print(f"API Request to / failed: {e}")
except Exception as e:
    print(f"An unexpected error occurred with /: {e}")
```

## Contributing

The Sentences Chunker is an open-source project that welcomes contributions from the community. Whether you're fixing bugs, improving documentation, adding new features, or sharing ideas, every contribution helps build a better tool for everyone.

If you're interested in contributing:

1. Fork the repository
2. Create a development environment with all dependencies
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a pull request

Happy Text Chunking!

---