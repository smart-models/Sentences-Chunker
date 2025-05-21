import re
import time
import asyncio
import logging
import tiktoken
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel


# Configure logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the logger
logger = logging.getLogger(__name__)

# Create a file handler for error logs
error_log_path = logs_dir / "errors.log"
file_handler = RotatingFileHandler(
    error_log_path,
    maxBytes=10485760,  # 10 MB
    backupCount=5,  # Keep 5 backup logs
    encoding="utf-8",
)

# Set the file handler to only log errors and critical messages
file_handler.setLevel(logging.ERROR)

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


# Pydantic models for response
class Chunk(BaseModel):
    """Represents a single text chunk with its token count and ID."""

    text: str
    token_count: int
    id: int


class ChunkingMetadata(BaseModel):
    """Metadata about the chunking process and results."""

    file: str
    n_chunks: int
    avg_tokens: int
    max_tokens: int
    min_tokens: int
    max_chunk_tokens: int
    overlap: int
    processing_time: float


class ChunkingResult(BaseModel):
    """Complete result of the chunking process."""

    chunks: List[Chunk]
    metadata: ChunkingMetadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    logger.info("Starting Text Chunker API...")
    # Any initialization can go here (e.g., loading models)
    yield
    # Cleanup code
    logger.info("Shutting down Text Chunker API...")


app = FastAPI(
    title="Text Chunker API",
    description="API for chunking text documents into smaller segments with control over token count and overlap",
    version="0.5.0",  # Incremented version number for the new functionality
    lifespan=lifespan,
)


def split_sentences_at_newline(doc: str) -> List[str]:
    """Split a document into sentences using newline characters (LF).

    This is a simple splitting method that treats each line as a separate sentence.
    It splits the text at newline characters and returns each line as a sentence.

    Args:
        doc (str): The input document text to be split into sentences.

    Returns:
        List[str]: A list of sentences, with each sentence stripped of leading/trailing whitespace.

    Raises:
        ValueError: If doc is None or not a string
        RuntimeError: If there's an error while processing the document
    """
    # Input validation
    if doc is None:
        logging.error("Input document is None")
        raise ValueError("Input document cannot be None")

    if not isinstance(doc, str):
        logging.error(f"Input document must be a string, got {type(doc)}")
        raise ValueError(f"Input document must be a string, got {type(doc)}")

    # Handle empty string case
    if not doc.strip():
        logging.warning("Empty document provided to split_sentences_at_newline")
        return []

    try:
        # Split the document at newline characters
        lines = doc.split("\n")

        # Strip whitespace from each line and filter out empty lines
        sentences = [line.strip() for line in lines if line.strip()]

        return sentences
    except Exception as e:
        logging.error(f"Unexpected error in split_sentences_at_newline: {str(e)}")
        raise RuntimeError(f"Error processing document: {str(e)}")


def split_into_sentences(doc: str) -> List[str]:
    """Split a document into sentences using regex pattern matching.

    Args:
        doc (str): The input document text to be split into sentences.

    Returns:
        List[str]: A list of sentences, with each sentence stripped of leading/trailing whitespace.

    Note:
        The function handles common edge cases like:
        - Titles (Mr., Mrs., Dr., etc.)
        - Common abbreviations (i.e., e.g., etc.)
        - Decimal numbers
        - Ellipsis
        - Quotes and brackets

    Raises:
        ValueError: If doc is None or not a string
        RuntimeError: If regex pattern matching fails
    """
    # Input validation
    if doc is None:
        logging.error("Input document is None")
        raise ValueError("Input document cannot be None")

    if not isinstance(doc, str):
        logging.error(f"Input document must be a string, got {type(doc)}")
        raise ValueError(f"Input document must be a string, got {type(doc)}")

    # Handle empty string case
    if not doc.strip():
        logging.warning("Empty document provided to split_into_sentences")
        return []

    # Define a pattern that looks for sentence boundaries but doesn't include them in the split
    # Instead of splitting directly at punctuation, we'll look for patterns that indicate sentence endings
    pattern = r"""
        # Match sentence ending punctuation followed by space and capital letter
        # Negative lookbehind for titles, abbreviations, initials, numbers, etc.
        (?<![A-Z][a-z]\.)          # Avoid splitting abbreviations like U.S.A.
        (?<!\s[A-Z]\.)             # Avoid splitting single initials like D. in Christopher D. Manning
        (?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ms\.) # Avoid splitting titles
        (?<!i\.e\.)(?<!e\.g\.)(?<!vs\.)(?<!etc\.)(?<!et al\.)             # Avoid splitting common abbreviations
        (?<!\d\.)(?<!\.\d)         # Avoid splitting decimal numbers or numbered lists
        (?<!\.\.\..)                # Avoid splitting ellipsis
        [\.!?]                    # Match sentence ending punctuation (. ! ?)
        \s+                        # Match one or more whitespace characters
        (?=[A-Z])                  # Positive lookahead for a capital letter (start of the next sentence)
    """

    try:
        # Find all positions where we should split
        split_positions = []
        for match in re.finditer(pattern, doc, re.VERBOSE):
            # Split after the punctuation and space
            split_positions.append(match.end())

        # Use the positions to extract sentences
        sentences = []
        start = 0
        for pos in split_positions:
            if pos > start:
                sentences.append(doc[start:pos].strip())
                start = pos

        # Add the last sentence if there's remaining text
        if start < len(doc):
            sentences.append(doc[start:].strip())

        # Filter out empty sentences
        return [s for s in sentences if s]
    except re.error as e:
        logging.error(f"Regex pattern error in split_into_sentences: {str(e)}")
        raise RuntimeError(f"Failed to parse document with regex: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error in split_into_sentences: {str(e)}")
        raise RuntimeError(f"Error processing document: {str(e)}")


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a text string using the specified encoding.

    Args:
        text (str): Text to count tokens for
        encoding_name (str): Name of the tiktoken encoding to use (default: cl100k_base)

    Returns:
        int: Number of tokens in the text

    Raises:
        ValueError: If the encoding name is invalid or not supported
        RuntimeError: If there's an issue with tokenization
    """
    try:
        if not text:
            return 0

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except KeyError:
        logging.error(f"Invalid encoding name: {encoding_name}")
        raise ValueError(f"Invalid or unsupported encoding name: '{encoding_name}'")
    except Exception as e:
        logging.error(f"Error during token counting: {str(e)}")
        raise RuntimeError(f"Failed to count tokens: {str(e)}")


def chunk_by_token_limit(
    sentence_data: List[Dict[str, Any]], max_chunk_tokens: int = 800, overlap: int = 0
) -> List[Dict[str, Any]]:
    """
    Groups sentences into chunks not exceeding the max token limit, with optional sentence overlap.

    Sentences are processed in order (from first to last) and concatenated until
    adding another sentence would exceed the token limit. Then a new chunk is started.
    Each chunk (except the first) can start with the last 'overlap' sentences from the previous chunk.

    Args:
        sentence_data (list of dict): Each dict has 'text', 'token_count', 'id' keys.
        max_chunk_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of sentences to overlap between consecutive chunks.

    Returns:
        list of dict: Each dict has 'text', 'token_count', and 'sentence_ids' keys.

    Raises:
        ValueError: If parameters are invalid or overlap is too large
        TypeError: If input types are incorrect
        RuntimeError: For other processing errors
    """
    # Validate input types
    if not isinstance(sentence_data, list):
        logging.error(f"sentence_data must be a list, got {type(sentence_data)}")
        raise TypeError(f"sentence_data must be a list, got {type(sentence_data)}")

    if not isinstance(max_chunk_tokens, int):
        logging.error(
            f"max_chunk_tokens must be an integer, got {type(max_chunk_tokens)}"
        )
        raise TypeError(
            f"max_chunk_tokens must be an integer, got {type(max_chunk_tokens)}"
        )

    if not isinstance(overlap, int):
        logging.error(f"overlap must be an integer, got {type(overlap)}")
        raise TypeError(f"overlap must be an integer, got {type(overlap)}")

    # Validate parameter values
    if max_chunk_tokens <= 0:
        logging.error(f"max_chunk_tokens must be positive, got {max_chunk_tokens}")
        raise ValueError(f"max_chunk_tokens must be positive, got {max_chunk_tokens}")

    if overlap < 0:
        logging.error(f"overlap must be non-negative, got {overlap}")
        raise ValueError("Overlap must be non-negative.")

    # Validate sentence_data contents
    if not sentence_data:
        logging.warning("Empty sentence_data provided to chunk_by_token_limit")
        return []

    try:
        # Validate each sentence item has the required keys
        for i, item in enumerate(sentence_data):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Item at index {i} must be a dictionary, got {type(item)}"
                )

            for key in ["text", "token_count", "id"]:
                if key not in item:
                    raise ValueError(f"Item at index {i} missing required key: '{key}'")

            if not isinstance(item["token_count"], int):
                raise TypeError(
                    f"token_count at index {i} must be an integer, got {type(item['token_count'])}"
                )

        chunks = []
        n = len(sentence_data)
        start_idx = 0

        while start_idx < n:
            current_chunk = []
            current_tokens = 0
            current_ids = []

            # Determine the starting index for this chunk (accounting for overlap)
            # For the first chunk, overlap is ignored
            if chunks:
                overlap_start = max(0, start_idx - overlap)
                overlap_sentences = sentence_data[overlap_start:start_idx]
                overlap_tokens = sum(item["token_count"] for item in overlap_sentences)
                if overlap_tokens > max_chunk_tokens:
                    raise ValueError(
                        f"Overlap too large: overlapping sentences alone ({overlap_tokens} tokens) exceed max_chunk_tokens ({max_chunk_tokens})"
                    )
                for item in overlap_sentences:
                    current_chunk.append(item["text"])
                    current_tokens += item["token_count"]
                    current_ids.append(item["id"])

            # Fill the chunk up to the token limit (do not re-add overlap sentences)
            idx = start_idx
            while (
                idx < n
                and current_tokens + sentence_data[idx]["token_count"]
                <= max_chunk_tokens
            ):
                current_chunk.append(sentence_data[idx]["text"])
                current_tokens += sentence_data[idx]["token_count"]
                current_ids.append(sentence_data[idx]["id"])
                idx += 1

            if current_chunk:
                chunks.append(
                    {
                        "text": " ".join(current_chunk),
                        "token_count": current_tokens,
                        "sentence_ids": current_ids,
                    }
                )

            # Move start_idx forward, but overlap with previous chunk
            if idx == start_idx:
                # This means a single sentence is too large to fit even with overlap
                sentence_tokens = sentence_data[start_idx]["token_count"]
                raise ValueError(
                    f"Cannot create chunk: sentence at index {start_idx} has {sentence_tokens} tokens, exceeding max_chunk_tokens ({max_chunk_tokens})"
                )
            start_idx = idx

        return chunks
    except (ValueError, TypeError) as e:
        # Re-raise these as they carry specific error information
        logging.error(f"Error in chunk_by_token_limit: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in chunk_by_token_limit: {str(e)}")
        raise RuntimeError(f"Error processing chunks: {str(e)}")


async def _find_min_max_tokens(
    sentence_data: List[Dict[str, Any]], max_sent_tok: int, overlap: int
) -> int:
    """
    Find the minimum token limit required for chunking with the given overlap.

    Uses binary search to find the smallest value of max_chunk_tokens
    that will successfully chunk the text with the specified overlap.

    Args:
        sentence_data: List of dicts with sentence info (text, token_count, id)
        max_sent_tok: Maximum number of tokens in any single sentence
        overlap: Number of sentences to overlap between chunks

    Returns:
        int: Minimum value of max_chunk_tokens that works with the given overlap
    """
    # Define lower and upper bounds
    lower_bound = max_sent_tok  # Cannot be less than the largest sentence
    # Adjust upper bound based on overlap - more generous for higher overlap values
    estimated_overlap_tokens = overlap * max_sent_tok if sentence_data else 0
    # Add a safety buffer
    upper_bound = max(lower_bound, estimated_overlap_tokens) + 250

    # The minimum value we've successfully chunked with - initialized higher
    min_success = upper_bound + 1

    # Counter to limit iterations
    iterations = 0
    max_iterations = 20  # Aumentato per casi più complessi

    logger.debug(
        f"Starting binary search for overlap {overlap}: range [{lower_bound}, {upper_bound}]"
    )

    # Binary search to find minimum working value
    found_success = False
    while lower_bound <= upper_bound and iterations < max_iterations:
        iterations += 1
        mid = (lower_bound + upper_bound) // 2

        if mid == 0:  # Evita di testare con max_tokens=0
            lower_bound = 1
            continue

        logger.debug(f"Iteration {iterations}: Testing mid={mid}")

        try:
            # Try to create chunks with the current mid value
            # We don't need to store the result, just check if it works without errors
            # Use asyncio.shield to protect against cancellation during chunking
            await asyncio.shield(
                asyncio.to_thread(chunk_by_token_limit, sentence_data, mid, overlap)
            )
            # If successful, this could be our answer, but we want the minimum
            logger.debug(f"Iteration {iterations}: Success with mid={mid}")
            min_success = mid
            found_success = True
            # Check if we can go lower
            upper_bound = mid - 1
        except ValueError as e:
            # If chunking fails, we need a higher value
            logger.debug(
                f"Iteration {iterations}: Failed with mid={mid} (ValueError: {e}). Increasing lower bound."
            )
            lower_bound = mid + 1
        except Exception as e:
            # Catch unexpected errors during the chunking test
            logger.error(
                f"Unexpected error during chunking test in _find_min_max_tokens (mid={mid}, overlap={overlap}): {str(e)}",
                exc_info=True,
            )
            # Treat unexpected errors as failure for this 'mid' value
            lower_bound = mid + 1

    logger.debug(
        f"Binary search finished after {iterations} iterations. Min success found: {min_success if found_success else 'None'}"
    )

    # If no successful value was found
    if not found_success:
        logger.error(
            f"Binary search failed to find any working max_chunk_tokens for overlap {overlap} within {max_iterations} iterations. Range was [{max_sent_tok}, {upper_bound} initially]."
        )
        # Fallback: return a value likely larger than needed
        return max_sent_tok + 100

    # Verify the final result (min_success) as it might be the last successful 'mid'
    try:
        logger.debug(f"Final verification with min_success={min_success}")
        await asyncio.shield(
            asyncio.to_thread(chunk_by_token_limit, sentence_data, min_success, overlap)
        )
        logger.debug(f"Final verification successful for {min_success}")
        return min_success
    except ValueError:
        # Add a small safety margin if verification fails
        logger.warning(
            f"Verification failed for min_success={min_success}. Returning {min_success + 5} as a safety measure."
        )
        return min_success + 5  # Add a small safety margin
    except Exception as e:
        # Return a safe fallback for unexpected verification errors
        logger.error(
            f"Unexpected error during final verification in _find_min_max_tokens (min_success={min_success}): {str(e)}",
            exc_info=True,
        )
        return max_sent_tok + 100


def _split_text_into_sentences(text: str, use_newline_splitting: bool) -> List[str]:
    """
    Helper function to split text into sentences based on the selected method.

    Args:
        text (str): The input text to be split into sentences.
        use_newline_splitting (bool): If True, splits text at newline characters;
                                      if False, uses regex pattern matching.

    Returns:
        List[str]: A list of sentences extracted from the input text.

    Raises:
        HTTPException: If an error occurs during sentence splitting.
    """
    try:
        if use_newline_splitting:
            sentences = split_sentences_at_newline(text)
            logger.info(
                f"Document split into {len(sentences)} sentences using newline splitting"
            )
        else:
            sentences = split_into_sentences(text)
            logger.info(
                f"Document split into {len(sentences)} sentences using regex pattern matching"
            )
        return sentences
    except (ValueError, RuntimeError) as e:
        logger.error(f"Error splitting document into sentences: {str(e)}")
        # Re-raise as HTTPException for the endpoints to catch
        raise HTTPException(
            status_code=400, detail=f"Error splitting document: {str(e)}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during sentence splitting: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during sentence splitting.",
        )


@app.get("/", tags=["Status"])
async def health_check():
    """Check if the API is running properly."""
    return {"status": "healthy", "version": app.version}


@app.post("/file-chunker", response_model=ChunkingResult, tags=["Chunking"])
async def file_chunker(
    file: UploadFile = File(...),
    max_chunk_tokens: int = Query(
        800, description="Maximum number of tokens per chunk", gt=0
    ),
    overlap: int = Query(
        0, description="Number of sentences to overlap between consecutive chunks", ge=0
    ),
    use_newline_splitting: bool = Query(
        False,
        description="If True, split text at newline characters; if False, use regex pattern matching",
    ),
):
    """
    Chunks a text file into smaller segments based on token limits with optional sentence overlap.

    Args:
        file: An uploaded text file
        max_chunk_tokens: Maximum number of tokens per chunk
        overlap: Number of sentences to overlap between consecutive chunks

    Returns:
        ChunkingResult: Contains the generated chunks and metadata about the chunking process
    """
    # Parameter validation first
    if max_chunk_tokens <= 0:
        error_msg = f"max_chunk_tokens must be positive, got {max_chunk_tokens}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    if overlap < 0:
        error_msg = f"overlap must be non-negative, got {overlap}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    if file is None:
        logger.error("File is required but was not provided")
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Log file processing attempt
        logger.info(
            f"Processing file: {file.filename} (size: {file.size if hasattr(file, 'size') else 'unknown'})"
        )

        # Check file size (optional, limit to 10MB)
        file_size_limit = 10 * 1024 * 1024  # 10MB
        if hasattr(file, "size") and file.size > file_size_limit:
            logger.warning(
                f"File {file.filename} exceeds size limit of {file_size_limit} bytes"
            )
            raise HTTPException(
                status_code=413, detail="File too large. Maximum allowed size is 10MB."
            )

        # Start timing
        start_time = time.time()

        # Check file extension (optional)
        file_ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
        allowed_extensions = {"txt", "md"}

        if file_ext not in allowed_extensions:
            logger.warning(f"Unsupported file format: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}",
            )

        # Read file content with timeout protection
        try:
            # Read file content
            file_content = await file.read()
        except asyncio.TimeoutError:
            logger.error(f"Timeout reading file {file.filename}")
            raise HTTPException(
                status_code=408,
                detail="Request timeout while reading file. Try with a smaller file.",
            )

        # Try to decode the content
        try:
            text = file_content.decode("utf-8")
        except AttributeError:
            # Already str
            text = file_content
        except UnicodeDecodeError:
            logger.error(f"Failed to decode file {file.filename} as UTF-8")
            raise HTTPException(
                status_code=400, detail="File must be a valid UTF-8 text file."
            )

        # Handle empty file case
        if not text.strip():
            logger.warning(f"Empty file content for {file.filename}")
            return ChunkingResult(
                chunks=[],
                metadata=ChunkingMetadata(
                    n_chunks=0,
                    avg_tokens=0,
                    max_tokens=0,
                    min_tokens=0,
                    max_chunk_tokens=max_chunk_tokens,
                    overlap=overlap,
                    processing_time=round(time.time() - start_time, 3),
                ),
            )

        # Step 1: Split the document using the helper function
        sentences = _split_text_into_sentences(text, use_newline_splitting)

        # Step 2: Count tokens for each sentence using default encoding (cl100k_base)
        sentence_data = []
        total_tokens = 0
        try:
            for i, sentence in enumerate(sentences):
                token_count = count_tokens(sentence)
                total_tokens += token_count
                sentence_data.append(
                    {"text": sentence, "token_count": token_count, "id": i + 1}
                )
            logger.info(f"Total token count: {total_tokens}")
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error counting tokens: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Error calculating tokens: {str(e)}"
            )

        # Step 3: Create chunks by token limit with optional overlap
        try:
            chunked_data = chunk_by_token_limit(
                sentence_data, max_chunk_tokens=max_chunk_tokens, overlap=overlap
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error during chunking: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            logger.error(f"Processing error during chunking: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Internal error during text chunking"
            )

        # Step 4: Create Chunk objects from the chunked data
        chunks = [
            Chunk(text=item["text"], token_count=item["token_count"], id=i + 1)
            for i, item in enumerate(chunked_data)
        ]

        # Step 5: Calculate metadata based on the chunks
        if chunks:
            chunk_token_counts = [chunk.token_count for chunk in chunks]
            processing_time = round(time.time() - start_time, 3)
            metadata = ChunkingMetadata(
                file=file.filename,  # Aggiungi il nome del file ai metadati
                n_chunks=len(chunks),
                avg_tokens=round(sum(chunk_token_counts) / len(chunk_token_counts))
                if chunk_token_counts
                else 0,
                max_tokens=max(chunk_token_counts) if chunk_token_counts else 0,
                min_tokens=min(chunk_token_counts) if chunk_token_counts else 0,
                max_chunk_tokens=max_chunk_tokens,
                overlap=overlap,
                processing_time=processing_time,
            )
        else:
            metadata = ChunkingMetadata(
                file=file.filename,  # Aggiungi il nome del file ai metadati
                n_chunks=0,
                avg_tokens=0,
                max_tokens=0,
                min_tokens=0,
                max_chunk_tokens=max_chunk_tokens,
                overlap=overlap,
                processing_time=round(time.time() - start_time, 3),
            )

        # Step 6: Create and return the final result
        result = ChunkingResult(chunks=chunks, metadata=metadata)

        logger.info(
            f"Successfully processed file {file.filename}: "
            f"Created {metadata.n_chunks} chunks, "
            f"Average tokens per chunk: {metadata.avg_tokens}, "
            f"Range: {metadata.min_tokens} to {metadata.max_tokens} tokens"
        )
        return result

    except UnicodeDecodeError:
        logger.error(f"Failed to decode file {file.filename} as UTF-8")
        raise HTTPException(
            status_code=400,
            detail="Unable to decode file. Please ensure the file is a valid text file with UTF-8 encoding.",
        )
    except HTTPException:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    except asyncio.CancelledError:
        logger.error(f"Request for file {file.filename} was cancelled")
        raise HTTPException(status_code=499, detail="Request cancelled")
    except MemoryError:
        logger.critical(f"Out of memory when processing file {file.filename}")
        raise HTTPException(
            status_code=500,
            detail="Server out of memory. File is too large to process.",
        )
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(
            f"Error processing file {file.filename}: {error_type} - {error_msg}",
            exc_info=True,
        )

        # Return a safe error message without exposing system details
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the file: {error_type}. Please check the logs for more information.",
        )


@app.post("/adaptive-file-chunking", response_model=ChunkingResult, tags=["Chunking"])
async def adaptive_file_chunking(
    file: UploadFile = File(...),
    overlap: int = Query(
        ...,
        description="Number of sentences to overlap (must be 0, 1, 2, or 3)",
        ge=0,
        le=3,
    ),
    use_newline_splitting: bool = Query(
        False,
        description="If True, split text at newline characters; if False, use regex pattern matching",
    ),
):
    """
    Chunks a text file into smaller segments using the minimum required token limit for the specified overlap.

    This endpoint calculates the minimum value of max_chunk_tokens needed for successful chunking
    with the specified overlap, then performs the chunking using that value. This ensures the
    densest possible chunks while maintaining the desired overlap.

    Args:
        file: An uploaded text file in UTF-8 format
        overlap: Number of sentences to overlap between consecutive chunks (0, 1, 2, or 3)

    Returns:
        ChunkingResult: Contains the generated chunks and metadata about the chunking process
    """

    if file is None:
        logger.error("File is required but was not provided")
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Log file processing attempt
        logger.info(
            f"Processing file with min tokens: {file.filename} (size: {file.size if hasattr(file, 'size') else 'unknown'})"
        )

        # Check file size (limit to 10MB)
        file_size_limit = 10 * 1024 * 1024  # 10MB
        if hasattr(file, "size") and file.size > file_size_limit:
            logger.warning(
                f"File {file.filename} exceeds size limit of {file_size_limit} bytes"
            )
            raise HTTPException(
                status_code=413, detail="File too large. Maximum allowed size is 10MB."
            )

        # Start timing
        start_time = time.time()

        # Check file extension
        file_ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
        allowed_extensions = {"txt", "md"}

        if file_ext not in allowed_extensions:
            logger.warning(f"Unsupported file format: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}",
            )

        # Read file content with timeout protection
        try:
            # Read file content
            file_content = await file.read()
        except asyncio.TimeoutError:
            logger.error(f"Timeout reading file {file.filename}")
            raise HTTPException(
                status_code=408,
                detail="Request timeout while reading file. Try with a smaller file.",
            )

        # Try to decode the content
        try:
            text = file_content.decode("utf-8")
        except AttributeError:
            # Already str
            text = file_content
        except UnicodeDecodeError:
            logger.error(f"Failed to decode file {file.filename} as UTF-8")
            raise HTTPException(
                status_code=400, detail="File must be a valid UTF-8 text file."
            )

        # Handle empty file case
        if not text.strip():
            logger.warning(f"Empty file content for {file.filename}")
            return ChunkingResult(
                chunks=[],
                metadata=ChunkingMetadata(
                    file=file.filename,  # Aggiungi il nome del file ai metadati
                    n_chunks=0,
                    avg_tokens=0,
                    max_tokens=0,
                    min_tokens=0,
                    max_chunk_tokens=0,
                    overlap=overlap,
                    processing_time=round(time.time() - start_time, 3),
                ),
            )

        # Step 1: Split the document using the helper function
        sentences = _split_text_into_sentences(text, use_newline_splitting)

        # Step 2: Count tokens for each sentence using default encoding (cl100k_base)
        sentence_data = []
        total_tokens = 0
        try:
            for i, sentence in enumerate(sentences):
                token_count = count_tokens(sentence)
                total_tokens += token_count
                sentence_data.append(
                    {"text": sentence, "token_count": token_count, "id": i + 1}
                )
            logger.info(f"Total token count: {total_tokens}")

            # Get maximum sentence token count
            max_sent_tok = (
                max([item["token_count"] for item in sentence_data])
                if sentence_data
                else 0
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error counting tokens: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Error calculating tokens: {str(e)}"
            )

        # Step 3: Calculate minimum required tokens for the specified overlap
        try:
            # Set a timeout for the calculation
            timeout_seconds = 60  # Longer timeout for this operation

            # Find minimum working max_chunk_tokens for the specified overlap
            min_required_tokens = await asyncio.wait_for(
                _find_min_max_tokens(sentence_data, max_sent_tok, overlap),
                timeout=timeout_seconds,
            )

            logger.info(
                f"Calculated minimum required tokens: {min_required_tokens} for overlap {overlap}"
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout ({timeout_seconds}s) calculating minimum token limit for file {file.filename}"
            )
            raise HTTPException(
                status_code=408,
                detail="Request timeout while calculating minimum token limit. Try with a smaller file or reduce overlap.",
            )
        except Exception as e:
            logger.error(f"Error calculating minimum token limit: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Error calculating minimum token limit"
            )

        # Step 4: Create chunks using the calculated minimum token limit
        try:
            chunked_data = chunk_by_token_limit(
                sentence_data, max_chunk_tokens=min_required_tokens, overlap=overlap
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error during chunking: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            logger.error(f"Processing error during chunking: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Internal error during text chunking"
            )

        # Step 5: Create Chunk objects from the chunked data
        chunks = [
            Chunk(text=item["text"], token_count=item["token_count"], id=i + 1)
            for i, item in enumerate(chunked_data)
        ]

        # Step 6: Calculate metadata based on the chunks
        if chunks:
            chunk_token_counts = [chunk.token_count for chunk in chunks]
            processing_time = round(time.time() - start_time, 3)
            metadata = ChunkingMetadata(
                file=file.filename,  # Aggiungi il nome del file ai metadati
                n_chunks=len(chunks),
                avg_tokens=round(sum(chunk_token_counts) / len(chunk_token_counts))
                if chunk_token_counts
                else 0,
                max_tokens=max(chunk_token_counts) if chunk_token_counts else 0,
                min_tokens=min(chunk_token_counts) if chunk_token_counts else 0,
                max_chunk_tokens=min_required_tokens,  # Use the calculated value
                overlap=overlap,
                processing_time=processing_time,
            )
        else:
            metadata = ChunkingMetadata(
                file=file.filename,  # Aggiungi il nome del file ai metadati
                n_chunks=0,
                avg_tokens=0,
                max_tokens=0,
                min_tokens=0,
                max_chunk_tokens=min_required_tokens,  # Use the calculated value
                overlap=overlap,
                processing_time=round(time.time() - start_time, 3),
            )

        # Step 7: Create and return the final result
        result = ChunkingResult(chunks=chunks, metadata=metadata)

        logger.info(
            f"Successfully processed file {file.filename} with min tokens approach: "
            f"Created {metadata.n_chunks} chunks using min_required_tokens={min_required_tokens}, "
            f"Average tokens per chunk: {metadata.avg_tokens}, "
            f"Range: {metadata.min_tokens} to {metadata.max_tokens} tokens"
        )
        return result

    except UnicodeDecodeError:
        logger.error(f"Failed to decode file {file.filename} as UTF-8")
        raise HTTPException(
            status_code=400,
            detail="Unable to decode file. Please ensure the file is a valid text file with UTF-8 encoding.",
        )
    except HTTPException:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    except asyncio.CancelledError:
        logger.error(f"Request for file {file.filename} was cancelled")
        raise HTTPException(status_code=499, detail="Request cancelled")
    except MemoryError:
        logger.critical(f"Out of memory when processing file {file.filename}")
        raise HTTPException(
            status_code=500,
            detail="Server out of memory. File is too large to process.",
        )
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(
            f"Error processing file {file.filename} with min tokens: {error_type} - {error_msg}",
            exc_info=True,
        )

        # Return a safe error message without exposing system details
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the file: {error_type}. Please check the logs for more information.",
        )
