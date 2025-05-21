**1. Purpose of the Code**

Imagine you have a long text document, like an article, a report, or even raw logs. The **Sentences Chunker** provides a web service (an API) that intelligently breaks this long text down into smaller, manageable pieces, called "chunks." The main goal is to segment the text while respecting natural sentence boundaries and user-defined size limits.

Unlike some chunkers that might just cut text at arbitrary points, this tool aims to:

*   **Preserve Sentence Integrity**: It primarily splits text *between* sentences, trying to keep whole sentences intact within chunks.
*   **Control Chunk Size**: It ensures that each chunk (or most chunks) stays below a maximum size limit you specify, measured in "tokens" (which are like words or parts of words, counted by `tiktoken`).
*   **Use Advanced Sentence Segmentation**: The tool uses WTPSplit's SaT (Segment any Text) models for intelligent, multilingual sentence boundary detection.
*   **Manage Context with Overlap**: You can specify a number of sentences to overlap between consecutive chunks, helping to maintain context if the chunks are processed sequentially by another system.
*   **Provide Detailed Metadata**: The service returns comprehensive statistics about both the input sentences and the output chunks, enabling better analysis of the chunking process.

Essentially, it's a robust text splitter designed to prepare text for further processing by other AI models (like Large Language Models for RAG systems) or any application that benefits from well-structured, size-controlled text segments.

**2. Input(s) it Takes**

To use this service, you send it an HTTP request (typically a `POST` request) with the following:

*   **A Text File (`file` parameter):** You must upload a file containing the text you want to chunk. The API supports files ending in `.txt` (plain text) or `.md` (Markdown).
*   **Maximum Chunk Tokens (`max_chunk_tokens` parameter - for `/file-chunker/` endpoint):**
    *   This is an *optional* integer (default is 800).
    *   You tell the API the maximum number of tokens allowed in each chunk. Must be greater than 0.
*   **Model Name (`model_name` parameter):**
    *   An *optional* string parameter specifying which SaT model to use for sentence segmentation.
    *   Default is "sat-12l-sm". Available options include various models with different layer counts (1l, 3l, 6l, 9l, 12l) and sizes (standard or small "sm").
*   **Split Threshold (`split_threshold` parameter):**
    *   An *optional* float (default is 0.5) between 0.0 and 1.0.
    *   Controls the confidence threshold for sentence boundary detection.
*   **Sentence Overlap (`overlap_sentences` parameter - for file-chunker endpoint):**
    *   An *optional* integer (default is 1).
    *   This specifies how many sentences from the end of the previous chunk should be included at the beginning of the current chunk.
    *   Can be 0 (no overlap) or any positive integer. Zero disables overlap completely.
*   **Strict Mode (`strict_mode` parameter):**
    *   An *optional* boolean (default is `False`).
    *   If `True`, the API will return an error if any chunk cannot strictly adhere to the token limit or overlap settings.

**3. Output(s) it Produces**

When the API successfully processes your file, it sends back a JSON response containing:

### For the `/file-chunker/` endpoint:

*   **A List of Chunks (`chunks` field):** This is the primary output. It's an array where each item represents a chunk of text. For each chunk, you get:
    *   `text`: The actual text content of the chunk.
    *   `token_count`: How many tokens (counted by `tiktoken`) are in this specific chunk.
    *   `id`: A simple number (1, 2, 3...) to identify the chunk's order.
    *   `overflow_details`: (Only in non-strict mode) Information about why limits were exceeded, if applicable.
*   **Metadata (`metadata` field):** This provides comprehensive information about the chunking process:
    *   File information: Name of the processed file
    *   Configuration: The token limit, overlap setting, model name, and split threshold used
    *   Input sentence statistics: Count, average/min/max token counts of sentences before chunking
    *   Output chunk statistics: Count, average/min/max token counts of the final chunks
    *   Processing time: Total time taken in seconds

### For the `/split-sentences/` endpoint:

*   **A List of Chunks (`chunks` field):** Each item represents a single sentence:
    *   `text`: The sentence text.
    *   `token_count`: Number of tokens in the sentence.
    *   `id`: Sequential ID of the sentence.
*   **Metadata (`metadata` field):** Provides statistics about the sentence splitting process.

**4. How it Achieves its Purpose (Logic and Algorithms)**

The API follows these main steps when you call one of its endpoints:

### For both `/file-chunker/` and `/split-sentences/` endpoints:

*   **Step 1: File Reception and Validation:**
    *   The API receives the uploaded file and parameters.
    *   It performs basic validation (e.g., file presence, valid parameter values, allowed extensions).
*   **Step 2: Text Preprocessing and Sentence Splitting:**
    *   The content of the file is read and decoded as UTF-8 text.
    *   The text is preprocessed to replace single line breaks with spaces while preserving paragraph breaks.
    *   The WTPSplit SaT model is used to intelligently split the text into sentences, using the specified model and split_threshold.

### For the `/split-sentences/` endpoint:

*   **Step 3: Sentence Tokenization and Output Preparation:**
    *   For each sentence, its token count is calculated using `tiktoken` with the `cl100k_base` encoding.
    *   The list of sentences with their token counts and IDs is formatted as the response.
    *   Metadata is calculated and included in the response.

### For the `/file-chunker/` endpoint:

*   **Step 3: Sentence Tokenization and Data Preparation:**
    *   For each sentence, its token count is calculated.
    *   This creates a list of sentence data, where each item contains the sentence text, its token count, and an ID.
*   **Step 4: Chunking by Token Limit:**
    *   The core chunking logic takes the sentence data and groups it into chunks based on the max_chunk_tokens parameter.
    *   It iterates through the sentences, accumulating them into a "current chunk."
    *   If overlap_sentences is greater than 0, and it's not the first chunk, it will prepend the specified number of sentences from the end of the previous chunk to the current chunk.
    *   It keeps adding sentences to the current chunk as long as the total token count stays below max_chunk_tokens.
    *   If strict_mode is enabled and limits cannot be respected (e.g., a single sentence exceeds the token limit), an error with detailed suggestions is returned.
    *   Otherwise, in non-strict mode, oversized chunks are created with warning details.
*   **Step 5: Formatting Output:**
    *   The list of generated chunks is prepared.
    *   Comprehensive metadata is calculated based on both the input sentences and the output chunks.
    *   The final FileChunkingResult is constructed and returned as a JSON response.

**5. Important Logic Flows or Data Transformations**

*   **File Bytes to Text String:** Raw bytes from the uploaded file are decoded into a UTF-8 string.
*   **Text Preprocessing:** Single line breaks are replaced with spaces while preserving paragraph breaks (double line breaks).
*   **WTPSplit Sentence Segmentation:** The advanced SaT model analyzes the text to intelligently identify sentence boundaries across multiple languages and edge cases.
*   **Sentence Data Structure:** Each sentence is augmented with its token count and an ID, forming a structured list.
*   **Robust Chunking Algorithm:** The `_chunk_sentences_by_token_limit` function handles the core logic of grouping sentences into chunks while respecting token limits and managing overlap.
*   **Strict Mode Error Handling:** When limits cannot be respected in strict mode, the StrictChunkingError provides detailed suggestions for parameter adjustments.
*   **Comprehensive Metadata Calculation:** Statistics are calculated for both input sentences and output chunks to provide insight into the chunking process.
*   **Model Caching with Expiration:** The SaT models are cached in memory with a timeout to improve performance while managing memory usage.
*   **Asynchronous Operations:** FastAPI's async capabilities are leveraged for file reading and processing to prevent blocking the server.
*   **GPU Acceleration:** The tool can use CUDA-enabled GPUs for faster model inference when available.

**6. API Endpoints**

### Health Check Endpoint (`GET /`)

Returns basic status information including API health, version, GPU availability, and default model configuration.

### Split Sentences Endpoint (`POST /split-sentences/`)

Splits a text document into individual sentences using WTPSplit's advanced segmentation.

### File Chunker Endpoint (`POST /file-chunker/`)

Chunks a text document into segments based on token limits and sentence overlap, with options for strict token limiting.