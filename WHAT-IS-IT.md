

**1. Purpose of the Code**

Imagine you have a long text document, like an article, a report, or even raw logs. The **Sentences Chunker** provides a web service (an API) that intelligently breaks this long text down into smaller, manageable pieces, called "chunks." The main goal is to segment the text while respecting natural sentence boundaries and user-defined size limits.

Unlike some chunkers that might just cut text at arbitrary points, this tool aims to:

*   **Preserve Sentence Integrity**: It primarily splits text *between* sentences, trying to keep whole sentences intact within chunks.
*   **Control Chunk Size**: It ensures that each chunk (or most chunks) stays below a maximum size limit you specify, measured in "tokens" (which are like words or parts of words, counted by `tiktoken`).
*   **Offer Flexible Splitting**: You can choose how sentences are initially identified: either by simple newline characters or by a more sophisticated regex-based method that understands common sentence structures.
*   **Manage Context with Overlap**: You can specify a number of sentences to overlap between consecutive chunks, helping to maintain context if the chunks are processed sequentially by another system.
*   **Adapt to Content (Adaptive Chunking)**: One of its key features allows it to automatically find the *smallest possible* maximum token limit that will still allow the entire document to be chunked with your desired sentence overlap. This creates the densest possible chunks.

Essentially, it's a robust text splitter designed to prepare text for further processing by other AI models (like Large Language Models for RAG systems) or any application that benefits from well-structured, size-controlled text segments.

**2. Input(s) it Takes**

To use this service, you send it an HTTP request (typically a `POST` request) with the following:

*   **A Text File (`file` parameter):** You must upload a file containing the text you want to chunk. The API supports files ending in `.txt` (plain text) or `.md` (Markdown).
*   **Maximum Chunk Tokens (`max_chunk_tokens` parameter - for `/file-chunker/` endpoint):**
    *   This is an *optional* integer (default is 800).
    *   You tell the API the maximum number of tokens allowed in each chunk. Must be greater than 0.
*   **Sentence Overlap (`overlap` parameter - for both chunking endpoints):**
    *   For `/file-chunker/`: An *optional* integer (default is 0).
    *   For `/adaptive-file-chunking/`: A *required* integer (must be 1, 2, or 3).
    *   This specifies how many sentences from the end of the previous chunk should be included at the beginning of the current chunk. Must be non-negative.
*   **Use Newline Splitting (`use_newline_splitting` parameter - for both chunking endpoints):**
    *   An *optional* boolean (default is `False`).
    *   If `True`, the text is split into sentences based on newline characters (`\n`).
    *   If `False` (default), a more complex regular expression is used to identify sentence boundaries.

**3. Output(s) it Produces**

When the API successfully processes your file, it sends back a JSON response containing:

*   **A List of Chunks (`chunks` field):** This is the primary output. It's an array where each item represents a chunk of text. For each chunk, you get:
    *   `text`: The actual text content of the chunk.
    *   `token_count`: How many tokens (counted by `tiktoken`) are in this specific chunk.
    *   `id`: A simple number (1, 2, 3...) to identify the chunk's order.
*   **Metadata (`metadata` field):** This provides summary information about the chunking process:
    *   `file`: The name of the original file you uploaded.
    *   `n_chunks`: The total number of chunks created.
    *   `avg_tokens`: The average number of tokens per chunk.
    *   `max_tokens`: The token count of the largest *actual* chunk created.
    *   `min_tokens`: The token count of the smallest *actual* chunk created.
    *   `max_chunk_tokens`: The target maximum token limit that was used for chunking (either user-specified or adaptively determined).
    *   `overlap`: The sentence overlap setting that was used.
    *   `processing_time`: How long the whole process took in seconds.

**4. How it Achieves its Purpose (Logic and Algorithms)**

The API follows these main steps when you call one of its chunking endpoints (`/file-chunker/` or `/adaptive-file-chunking/`):

*   **Step 1: File Reception and Validation:**
    *   The API receives the uploaded file and parameters.
    *   It performs basic validation (e.g., file presence, valid parameter values, file size limits, allowed extensions).
*   **Step 2: Text Decoding and Sentence Splitting (`_split_text_into_sentences` internal function):**
    *   The content of the file is read and decoded as UTF-8 text.
    *   Based on the `use_newline_splitting` parameter, the text is split into an initial list of sentences:
        *   If `True`: Uses `split_sentences_at_newline`, which splits the document at newline characters.
        *   If `False`: Uses `split_into_sentences`, which employs a regular expression designed to identify sentence boundaries more robustly, considering things like titles (Mr., Dr.), abbreviations, and decimal numbers.
*   **Step 3: Sentence Tokenization and Data Preparation:**
    *   For each sentence identified in Step 2, its token count is calculated using `tiktoken` (typically with the `cl100k_base` encoding used by many OpenAI models).
    *   This creates a list of sentence data, where each item contains the sentence text, its token count, and an ID.
*   **Step 4 (Only for `/adaptive-file-chunking/`): Determine Optimal `max_chunk_tokens` (`_find_min_max_tokens` internal function):**
    *   If you're using the `/adaptive-file-chunking/` endpoint, this step is crucial.
    *   The API performs a binary search to find the *minimum* possible value for `max_chunk_tokens` that will allow the entire document to be chunked successfully with the `overlap` you specified.
    *   It repeatedly tries to chunk the `sentence_data` (from Step 3) with different `max_chunk_tokens` values until it finds the smallest one that doesn't cause errors (like a single sentence being too large even with overlap). This ensures the densest possible packing of tokens into chunks.
*   **Step 5: Chunking by Token Limit (`chunk_by_token_limit` function):**
    *   This is the core chunking logic. It takes the `sentence_data` (from Step 3) and the `max_chunk_tokens` (either user-provided for `/file-chunker/` or determined in Step 4 for `/adaptive-file-chunking/`) and the `overlap` value.
    *   It iterates through the sentences, accumulating them into a "current chunk."
    *   If `overlap` is greater than 0, and it's not the first chunk, it will prepend the last `overlap` sentences from the *previous* chunk to the *current* chunk.
    *   It keeps adding sentences to the current chunk as long as the total token count of the chunk (including any overlapped sentences and the next sentence) does not exceed `max_chunk_tokens`.
    *   Once adding the next sentence would exceed the limit, the current chunk is finalized, and a new chunk begins (potentially starting with an overlap from the just-finalized chunk).
    *   If a single sentence (even after considering overlap) is larger than `max_chunk_tokens`, an error is raised, indicating that the `max_chunk_tokens` is too small for the content.
*   **Step 6: Formatting Output:**
    *   The list of generated chunks (each containing text and its token count) is prepared.
    *   The summary `metadata` is calculated based on the resulting chunks and the input parameters.
    *   The final `ChunkingResult` (containing `chunks` and `metadata`) is constructed and returned as a JSON response.

**5. Important Logic Flows or Data Transformations**

*   **File Bytes to Text String:** Raw bytes from the uploaded file are decoded into a UTF-8 string.
*   **Text String to List of Sentence Strings:** The primary text is transformed into an ordered list of individual sentences.
*   **Sentence Strings to Sentence Data (List of Dictionaries):** Each sentence string is augmented with its token count and an ID, forming a structured list.
*   **Sentence Data to Chunks (List of Dictionaries):** The `chunk_by_token_limit` function transforms the list of sentence data into a list of chunk data, where each chunk aggregates multiple sentences and has its own total token count.
*   **Binary Search for Optimal Token Limit (Adaptive Endpoint):** The `_find_min_max_tokens` function iteratively calls a version of the chunking logic to narrow down the best `max_chunk_tokens` value. This involves repeated "test chunking" operations.
*   **Error Handling & Logging:** The API includes extensive `try...except` blocks to handle potential errors (e.g., file decoding errors, invalid parameters, issues during chunking). It uses Python's `logging` module to record informational messages and detailed error messages (errors are also written to a rotating log file in a `logs` directory). This is crucial for diagnostics and understanding API behavior.
*   **Asynchronous Operations:** The API is built with FastAPI, which supports asynchronous request handling (`async def`). File reading (`await file.read()`) and the adaptive token limit search (`await asyncio.wait_for(_find_min_max_tokens, ...)`), in particular, are handled asynchronously to prevent blocking the server.

In summary, the Sentences Chunker provides a reliable and configurable way to segment text documents based on sentence structure and token limits, with an advanced option for optimizing chunk density. It's designed with clear separation of concerns: sentence splitting, token counting, and the core chunking logic.
```