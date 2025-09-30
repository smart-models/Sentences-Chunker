import re
import time
import logging
import tiktoken
import threading
import torch
import math
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager


# Valid SaT model names
VALID_SAT_MODELS = {
    "sat-1l",
    "sat-1l-sm",
    "sat-3l",
    "sat-3l-sm",
    "sat-6l",
    "sat-6l-sm",
    "sat-9l",
    "sat-12l",
    "sat-12l-sm",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration Constants
DEFAULT_SAT_MODEL_NAME = "sat-12l-sm"
DEFAULT_SAT_SPLIT_THRESHOLD = 0.5
TIKTOKEN_ENCODING = "cl100k_base"
CACHE_TIMEOUT = 3600  # Model cache timeout in seconds (1 hour)
SUGGESTION_SAFETY_MARGIN_PERCENT = 0.30  # 30% safety margin for suggestions


# Custom exception for strict mode chunking
class StrictChunkingError(ValueError):
    """Custom exception for errors during strict mode chunking.

    This exception is raised when strict mode is enabled and chunking limits cannot be respected.
    It includes detailed information about the failure and suggestions for parameters to resolve the issue.
    """

    def __init__(
        self,
        message: str,
        details: Dict[str, Any] = None,
        suggestions: Dict[str, Any] = None,
    ):
        super().__init__(message)
        self.details = details if details is not None else {}
        self.suggestions = suggestions if suggestions is not None else {}


# Pydantic models for response
class Chunk(BaseModel):
    """Represents a single text chunk with its token count and ID.

    A chunk is a piece of text that has been created by grouping one or more sentences
    while respecting token limits. In non-strict mode, a chunk might exceed the configured
    token limit, in which case it will include overflow_details explaining why.
    """

    text: str = Field(..., description="The actual text content of the chunk.")
    token_count: int = Field(..., description="Number of tokens in the chunk.")
    id: int = Field(..., description="Sequential ID of the chunk.")
    overflow_details: Optional[str] = Field(
        None,
        description="Details if the chunk exceeded configured limits but was processed in non-strict mode.",
    )


class ChunkingMetadata(BaseModel):
    """Metadata about the sentence splitting process and results."""

    n_sentences: int = Field(..., description="Total number of sentences generated.")
    split_threshold: float = Field(
        ..., description="Threshold used for sentence splitting."
    )
    avg_tokens_per_sentence: int = Field(
        ..., description="Average number of tokens per sentence."
    )
    max_tokens_in_sentence: int = Field(
        ..., description="Maximum number of tokens in any single sentence."
    )
    min_tokens_in_sentence: int = Field(
        ..., description="Minimum number of tokens in any single sentence."
    )
    sat_model_name: str = Field(
        ..., description="Name of the SaT model used for splitting."
    )
    source: str = Field(..., description="Name of the processed file.")
    processing_time: float = Field(
        ..., description="Total time taken for processing in seconds."
    )


class FileChunkingMetadata(BaseModel):
    """Metadata about the file chunking process and results."""

    configured_max_chunk_tokens: int = Field(
        ..., description="Maximum token limit per chunk for token chunker."
    )
    configured_overlap_sentences: int = Field(
        ..., description="Number of overlapping sentences between chunks."
    )
    n_input_sentences: int = Field(
        ..., description="Number of sentences before chunking."
    )
    avg_tokens_per_input_sentence: int = Field(
        ..., description="Average tokens per sentence before chunking."
    )
    max_tokens_in_input_sentence: int = Field(
        ..., description="Max tokens in a sentence before chunking."
    )
    min_tokens_in_input_sentence: int = Field(
        ..., description="Min tokens in a sentence before chunking."
    )
    n_chunks: int = Field(..., description="Total number of chunks generated.")
    avg_tokens_per_chunk: int = Field(..., description="Average tokens per chunk.")
    max_tokens_in_chunk: int = Field(..., description="Maximum tokens in any chunk.")
    min_tokens_in_chunk: int = Field(..., description="Minimum tokens in any chunk.")
    sat_model_name: str = Field(
        ..., description="Name of the SaT model used for splitting."
    )
    split_threshold: float = Field(
        ..., description="Threshold used for sentence splitting."
    )
    source: str = Field(..., description="Name of the processed file.")
    processing_time: float = Field(
        ..., description="Total time taken for processing in seconds."
    )


class ChunkingResult(BaseModel):
    """Results from the sentence splitting operation."""

    chunks: List[Chunk] = Field(..., description="List of generated sentences.")
    metadata: ChunkingMetadata = Field(
        ..., description="Metadata about the sentence splitting process."
    )


class FileChunkingResult(BaseModel):
    """Results from the file chunking operation."""

    chunks: List[Chunk] = Field(..., description="List of generated chunks.")
    metadata: FileChunkingMetadata = Field(
        ..., description="Metadata about the file chunking process."
    )


class SplitSentencesInput(BaseModel):
    """Input parameters for split sentences endpoint."""

    model_name: str = Field(
        default=DEFAULT_SAT_MODEL_NAME,
        description="The SaT model to use for sentence segmentation",
    )
    split_threshold: float = Field(
        default=DEFAULT_SAT_SPLIT_THRESHOLD,
        description="Threshold value for sentence splitting (confidence score for sentence boundaries)",
        ge=0.0,
        le=1.0,
    )


class FileChunkerInput(BaseModel):
    """Input parameters for file chunking endpoint."""

    model_name: str = Field(
        default=DEFAULT_SAT_MODEL_NAME,
        description="The SaT model to use for sentence segmentation",
    )
    split_threshold: float = Field(
        default=DEFAULT_SAT_SPLIT_THRESHOLD,
        description="Threshold value for sentence splitting (confidence score for sentence boundaries)",
        ge=0.0,
        le=1.0,
    )
    max_chunk_tokens: int = Field(
        default=500, description="Maximum number of tokens per final chunk", gt=0
    )
    overlap_sentences: int = Field(
        default=1,
        description="Number of sentences to overlap between consecutive chunks",
        ge=0,
    )
    strict_mode: bool = Field(
        default=False,
        description="If True, an error is returned if any chunk cannot strictly adhere to token/overlap limits",
    )


# Create a singleton for model caching with expiration
_model_cache = {}
_model_last_used = {}
_model_lock = threading.RLock()  # Thread-safe lock for model cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    logger.info("Starting Text Chunker API...")

    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    logger.info(f"Models directory: {models_dir.absolute()}")

    # Initialize the SaT model during startup
    try:
        logger.info(
            f"Loading WTPSplit model {DEFAULT_SAT_MODEL_NAME} during application startup..."
        )
        _get_sat_model(DEFAULT_SAT_MODEL_NAME)
        logger.info("WTPSplit model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load WTPSplit model: {str(e)}")

    yield

    # Cleanup code
    logger.info("Shutting down Text Chunker API...")
    try:
        # Cleanup model cache
        with _model_lock:
            for model_name in list(_model_cache.keys()):
                if model_name in _model_cache:
                    # Move model to CPU before deletion to free GPU memory
                    if torch.cuda.is_available():
                        try:
                            _model_cache[model_name].cpu()
                        except Exception:
                            pass  # Model might not have a .cpu() method
                    del _model_cache[model_name]
                    logger.info(f"Removed model {model_name} from memory cache")

        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared during shutdown")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="Text Chunker API",
    description="API for chunking text documents into smaller segments with control over token count and overlap",
    version="0.6.6",
    lifespan=lifespan,
)


def _get_sat_model(model_name: str = DEFAULT_SAT_MODEL_NAME):
    """Get or initialize the SaT model with persistent disk caching.

    Args:
        model_name (str): Name of the SaT model to use

    Returns:
        SaT: The initialized SaT model instance.

    Raises:
        ImportError: If WTPSplit is not installed
        RuntimeError: If there's an error initializing the model
    """
    current_time = time.time()
    model_key = model_name  # Using the model name as the cache key

    with _model_lock:
        # Check for expired models first
        expired_models = [
            name
            for name, last_used in _model_last_used.items()
            if current_time - last_used > CACHE_TIMEOUT
        ]

        # Remove expired models from memory cache
        for name in expired_models:
            if (
                name in _model_cache and name != model_key
            ):  # Don't remove the one we're about to use
                logger.info(f"Removing expired model {name} from memory cache")
                del _model_cache[name]
                del _model_last_used[name]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Update or load the requested model
        if model_key not in _model_cache:
            try:
                # Lazy import to avoid loading the model until needed
                from wtpsplit import SaT

                # Create models directory if it doesn't exist
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)

                # Local path for the model
                local_model_path = models_dir / f"{model_key.replace('/', '_')}.pkl"

                if local_model_path.exists():
                    # Load from local storage
                    logger.info(f"Loading model from local storage: {local_model_path}")
                    try:
                        with open(local_model_path, "rb") as f:
                            _model_cache[model_key] = pickle.load(f)
                        logger.info(
                            f"SaT model {model_key} loaded from disk successfully"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load model from disk: {str(e)}, downloading fresh copy"
                        )
                        # If loading fails, download fresh
                        _model_cache[model_key] = SaT(model_key)
                        # Save the freshly downloaded model
                        with open(local_model_path, "wb") as f:
                            pickle.dump(_model_cache[model_key], f)
                        logger.info(
                            f"SaT model {model_key} downloaded and saved to disk"
                        )
                else:
                    # Download and save model
                    logger.info(
                        f"Downloading model {model_key} and saving to {local_model_path}"
                    )
                    _model_cache[model_key] = SaT(model_key)
                    # Save to disk for future use
                    with open(local_model_path, "wb") as f:
                        pickle.dump(_model_cache[model_key], f)
                    logger.info(f"SaT model {model_key} saved to disk successfully")

                # Use GPU if available for better performance
                if torch.cuda.is_available():
                    _model_cache[model_key].half().to("cuda")
                    logger.info("SaT model moved to GPU")
                else:
                    logger.info("SaT model kept on CPU")

            except ImportError:
                error_msg = "WTPSplit is not installed. Please install it with: pip install wtpsplit"
                logger.error(error_msg)
                raise ImportError(error_msg)
            except Exception as e:
                error_msg = f"Error initializing SaT model: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        # Update last used timestamp
        _model_last_used[model_key] = current_time

        return _model_cache[model_key]


def preprocess_text(text):
    """Replace single line breaks with spaces while preserving paragraph breaks.

    Args:
        text (str): Input text to preprocess

    Returns:
        str: Text with single line breaks replaced by spaces
    """
    # Replace single line breaks with spaces
    # Only preserves paragraph breaks (double line breaks)
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def split_sentences_NLP(
    doc: str,
    model_name: str = DEFAULT_SAT_MODEL_NAME,
    split_threshold: float = DEFAULT_SAT_SPLIT_THRESHOLD,
) -> List[str]:
    """Split a document into sentences using WTPSplit's advanced sentence segmentation.

    This function uses the Segment any Text (SaT) model from WTPSplit to intelligently
    split text into sentences, handling various edge cases and multiple languages.
    The model is loaded from disk cache if available and manages GPU memory efficiently.

    Args:
        doc (str): The input document text to be split into sentences.
        model_name (str): Name of the SaT model to use
        split_threshold (float): Threshold value for sentence splitting (0.0-1.0)

    Returns:
        List[str]: A list of sentences, with each sentence stripped of leading/trailing whitespace.

    Raises:
        ValueError: If doc is None or not a string or if split_threshold is not between 0 and 1
        RuntimeError: If there's an error while processing the document or initializing the model
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
        logging.warning("Empty document provided to split_sentences_NLP")
        return []

    try:
        # Validate split_threshold
        if not 0.0 <= split_threshold <= 1.0:
            raise ValueError(
                f"split_threshold must be between 0.0 and 1.0, got {split_threshold}"
            )

        # Get the cached model instance
        sat = _get_sat_model(model_name)

        # Ensure model is on the correct device for processing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and not next(sat.parameters()).is_cuda:
            sat.half().to(device)

        # Split the document into sentences
        sentences = sat.split(doc, threshold=split_threshold)

        # Move model back to CPU to free GPU memory for other operations
        if torch.cuda.is_available():
            sat.cpu()
            torch.cuda.empty_cache()

        # Filter out any empty strings that might result from the split
        return [s.strip() for s in sentences if s.strip()]

    except ImportError as e:
        # Already logged in get_sat_model
        raise e
    except Exception as e:
        # Clean GPU memory in case of error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        error_msg = f"Error in WTPSplit sentence segmentation: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def _round_up_to_nearest_multiple(number: float, multiple: int) -> int:
    """Round a number up to the nearest multiple of a given value.

    Args:
        number (float): The number to round up
        multiple (int): The multiple to round up to (if zero, rounds up to next integer)

    Returns:
        int: The number rounded up to nearest multiple
    """
    if multiple == 0:
        return math.ceil(number)
    return math.ceil(number / multiple) * multiple


def count_tokens(text: str, encoding_name: str = TIKTOKEN_ENCODING) -> int:
    """Count tokens in a text string using the specified encoding.

    This function uses the tiktoken library to count tokens according to the specified encoding.
    By default, it uses the 'cl100k_base' encoding which is compatible with many recent
    large language models. Token count is important for determining how text will be
    processed by models with token limits.

    Args:
        text: Text string to count tokens for
        encoding_name: Name of the tiktoken encoding to use (default: 'cl100k_base')

    Returns:
        int: Number of tokens in the text

    Raises:
        ValueError: If the specified encoding is not found in tiktoken
        RuntimeError: For other encoding or processing errors
    """
    if not text:
        return 0

    try:
        # Get the encoding for the specified model
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the text and count the tokens
        tokens = encoding.encode(text)
        return len(tokens)

    except KeyError:
        error_msg = f"Encoding '{encoding_name}' not found in tiktoken"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error counting tokens: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


async def _chunk_sentences_by_token_limit(
    sentences_data: List[Dict[str, Any]],
    max_chunk_tokens: int,
    configured_overlap_sentences: int,  # Renamed for clarity within this function
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Groups sentences into chunks based on token limits with optional overlap.

    This function takes a list of sentences with their token counts and groups them into
    larger chunks while respecting a maximum token limit. It can also create overlapping
    chunks by repeating a specified number of sentences between consecutive chunks.

    In strict mode, if any limit cannot be respected (e.g., a single sentence exceeds the
    token limit or the configured overlap would exceed the token limit), the function raises
    a StrictChunkingError with detailed suggestions for parameter adjustments.

    Args:
        sentences_data: List of dicts with 'text', 'token_count', and 'id' for each sentence
        max_chunk_tokens: Maximum tokens allowed per chunk
        configured_overlap_sentences: Number of sentences to overlap between chunks (0, 1, 2, or 3).
                                      Setting to 0 disables overlap.
        strict_mode: If True, raises StrictChunkingError when limits cannot be respected.
                     If False, creates oversized chunks with warning details.

    Returns:
        List of dicts with:
            - 'text': The combined text of all sentences in the chunk
            - 'token_count': Total tokens in the chunk
            - 'id': Sequential chunk identifier
            - 'original_sentence_ids': List of sentence IDs included in this chunk
            - 'overflow_details': (Only in non-strict mode) Information about why limits were exceeded

    Raises:
        ValueError: If parameters are invalid (e.g., negative max_chunk_tokens)
        TypeError: If input types are incorrect
        StrictChunkingError: When in strict mode and chunking limits cannot be respected.
                            Contains suggestions for parameter adjustments.
        RuntimeError: For other processing errors
    """
    # Validate input types
    if not isinstance(sentences_data, list):
        logging.error(f"sentences_data must be a list, got {type(sentences_data)}")
        raise TypeError(f"sentences_data must be a list, got {type(sentences_data)}")

    if not isinstance(max_chunk_tokens, int):
        logging.error(
            f"max_chunk_tokens must be an integer, got {type(max_chunk_tokens)}"
        )
        raise TypeError(
            f"max_chunk_tokens must be an integer, got {type(max_chunk_tokens)}"
        )

    if not isinstance(configured_overlap_sentences, int):
        logging.error(
            f"configured_overlap_sentences must be an integer, got {type(configured_overlap_sentences)}"
        )
        raise TypeError(
            f"configured_overlap_sentences must be an integer, got {type(configured_overlap_sentences)}"
        )

    # Validate parameter values
    if max_chunk_tokens <= 0:
        logging.error(f"max_chunk_tokens must be positive, got {max_chunk_tokens}")
        raise ValueError(f"max_chunk_tokens must be positive, got {max_chunk_tokens}")

    if configured_overlap_sentences < 0:
        logging.error(
            f"configured_overlap_sentences must be non-negative, got {configured_overlap_sentences}"
        )
        raise ValueError("configured_overlap_sentences must be non-negative.")

    # Validate sentences_data contents
    if not sentences_data:
        logging.warning(
            "Empty sentences_data provided to _chunk_sentences_by_token_limit"
        )
        return []

    try:
        # Validate each sentence item has the required keys
        for i, item in enumerate(sentences_data):
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
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_id_counter = 1
        # Flag to track if the current chunk being formed started with an oversized overlap
        current_chunk_overflow_reason_from_overlap = None

        for i, sentence in enumerate(sentences_data):
            # Determine if we need to include overflow details for the current chunk
            pending_overflow_details_for_chunk = (
                current_chunk_overflow_reason_from_overlap
            )

            # If adding this sentence would exceed the limit and we have something in the current chunk
            if current_chunk_sentences and (
                current_chunk_tokens + sentence["token_count"] > max_chunk_tokens
            ):
                # Finalize current chunk
                chunk_text = " ".join(s["text"] for s in current_chunk_sentences)
                chunk_to_add = {
                    "text": chunk_text,
                    "token_count": current_chunk_tokens,
                    "id": chunk_id_counter,
                    "original_sentence_ids": [s["id"] for s in current_chunk_sentences],
                }
                if pending_overflow_details_for_chunk and not strict_mode:
                    chunk_to_add["overflow_details"] = (
                        pending_overflow_details_for_chunk
                    )

                chunks.append(chunk_to_add)
                chunk_id_counter += 1
                current_chunk_overflow_reason_from_overlap = (
                    None  # Reset after using it
                )

                # Start new chunk with overlap
                if configured_overlap_sentences > 0 and current_chunk_sentences:
                    # Take last N sentences from previous chunk as overlap
                    overlap_start_index = max(
                        0, len(current_chunk_sentences) - configured_overlap_sentences
                    )
                    overlap_data = current_chunk_sentences[overlap_start_index:]
                    overlap_tokens_count = sum(s["token_count"] for s in overlap_data)

                    if overlap_tokens_count > max_chunk_tokens:
                        reason = (
                            f"Overlap of {len(overlap_data)} sentences ({overlap_tokens_count} tokens) "
                            f"exceeds max_chunk_tokens ({max_chunk_tokens})."
                        )
                        if strict_mode:
                            # STEP 1: Calculate suggested max_chunk_tokens value for this overlap scenario
                            # Start with the actual number of tokens in the overlap that failed
                            base_required_tokens_for_overlap = overlap_tokens_count

                            # Apply safety margin (e.g., 30%) to ensure the suggested value is robust
                            # This prevents suggesting a value that's just barely enough
                            tokens_for_overlap_with_margin = (
                                base_required_tokens_for_overlap
                                * (1 + SUGGESTION_SAFETY_MARGIN_PERCENT)
                            )

                            # Round up to nearest multiple of 100 for a cleaner, more user-friendly suggestion
                            # E.g., 523 tokens with margin becomes 600 rather than 523
                            suggested_max_t_for_overlap = _round_up_to_nearest_multiple(
                                tokens_for_overlap_with_margin, 100
                            )

                            # STEP 2: Calculate suggested overlap_sentences value
                            # Initialize with None - this will be updated if we find a viable value
                            suggested_overlap_s_val = None

                            # Get number of sentences in the current chunk - these are the sentences
                            # from which we'll try to create a smaller overlap
                            num_sentences_in_prev = len(current_chunk_sentences)

                            # Determine the maximum overlap we need to test
                            # We can't overlap more sentences than what the user configured or what's available
                            max_possible_overlap_to_test = min(
                                num_sentences_in_prev,  # Can't overlap more than we have
                                configured_overlap_sentences,  # Can't overlap more than configured
                            )

                            # Algorithm: Work backwards through overlap sizes to find maximum viable overlap
                            # Start at 0 (no viable overlap found yet)
                            max_allowable_s_for_current_max_t = 0

                            # Test each possible overlap size from 1 to max possible
                            for k_s in range(1, max_possible_overlap_to_test + 1):
                                # Calculate starting index for taking last k_s sentences
                                start_idx_k = num_sentences_in_prev - k_s

                                # Extract the subset of sentences for this overlap size
                                temp_overlap_k_subset = current_chunk_sentences[
                                    start_idx_k:
                                ]

                                # Count total tokens in this potential overlap
                                tokens_k = sum(
                                    s["token_count"] for s in temp_overlap_k_subset
                                )

                                # If this overlap fits within current max_chunk_tokens, it's viable
                                if tokens_k <= max_chunk_tokens:
                                    max_allowable_s_for_current_max_t = (
                                        k_s  # Update maximum viable overlap
                                    )
                                else:
                                    break  # Stop testing larger overlaps once we find one that's too large

                            # STEP 3: Determine what overlap value to suggest based on our analysis
                            # Case 1: If we found a viable overlap smaller than what failed, suggest that value
                            if max_allowable_s_for_current_max_t < len(overlap_data):
                                suggested_overlap_s_val = (
                                    max_allowable_s_for_current_max_t
                                )
                            # Case 2: If even 1 sentence overlap would be too much, suggest disabling overlap (0)
                            elif (
                                max_allowable_s_for_current_max_t == 0
                                and len(overlap_data) > 0
                            ):
                                suggested_overlap_s_val = 0
                            # Note: In other cases, suggested_overlap_s_val remains None

                            # STEP 4: Create a detailed, user-friendly message with options
                            # Format a message giving the user multiple options to resolve the issue
                            msg_overlap = (
                                f"Current settings (max_tokens: {max_chunk_tokens}, overlap: {len(overlap_data)}) are too restrictive for overlap. "
                                f"Consider these options:\n"
                                # Option 1: Always suggest increasing max_chunk_tokens as the primary solution
                                f"1. Increase 'max_chunk_tokens' to around {suggested_max_t_for_overlap}. This may allow your original 'overlap_sentences' or a higher value.\n"
                            )
                            # Option 2a: If we found a viable smaller overlap, suggest that specific value
                            if suggested_overlap_s_val is not None:
                                msg_overlap += f"2. With your current 'max_chunk_tokens' ({max_chunk_tokens}), try setting 'overlap_sentences' to {suggested_overlap_s_val}.\n"
                            # Option 2b: If no viable overlap was found, suggest a more general approach
                            else:  # If no viable overlap could be calculated
                                msg_overlap += "2. Alternatively, significantly reduce 'overlap_sentences' (possibly to 0) if you cannot increase 'max_chunk_tokens'.\n"
                            # Add footer to emphasize these are estimates
                            msg_overlap += (
                                "These are initial estimates to help proceed."
                            )

                            # STEP 5: Package all suggestions into a structured object
                            suggestions = {
                                "suggested_max_chunk_tokens": suggested_max_t_for_overlap,
                                "suggested_overlap_sentences": suggested_overlap_s_val,
                                "message": msg_overlap,
                            }
                            raise StrictChunkingError(
                                f"Strict mode: {reason}",
                                details={
                                    "type": "overlap_too_large",
                                    "current_overlap_sentences": len(overlap_data),
                                    "overlap_tokens": overlap_tokens_count,
                                    "current_max_chunk_tokens": max_chunk_tokens,
                                },
                                suggestions=suggestions,
                            )
                        else:
                            current_chunk_overflow_reason_from_overlap = reason
                            logging.warning(
                                f"Chunk (to be ID {chunk_id_counter}) will start with oversized overlap: {reason}"
                            )
                    current_chunk_sentences = overlap_data
                    current_chunk_tokens = overlap_tokens_count
                else:
                    current_chunk_sentences = []
                    current_chunk_tokens = 0

            # Add current sentence to chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence["token_count"]

            # Handle case where a single sentence exceeds max_chunk_tokens
            if (
                len(current_chunk_sentences) == 1
                and current_chunk_tokens > max_chunk_tokens
            ):
                reason = (
                    f"Single sentence (ID {sentence['id']}) with {sentence['token_count']} tokens "
                    f"exceeds max_chunk_tokens ({max_chunk_tokens})."
                )
                if strict_mode:
                    # SINGLE SENTENCE CASE: A single sentence exceeds the maximum token limit
                    # This case is simpler than the overlap case because there's only one solution: increase the token limit

                    # STEP 1: Calculate a suggested max_chunk_tokens value based on this sentence
                    # Start with the actual token count of the sentence
                    base_required_tokens = sentence["token_count"]

                    # Apply safety margin (e.g., 30%) to ensure the suggested value works robustly
                    # This prevents suggesting a value that's just barely enough
                    tokens_with_margin = base_required_tokens * (
                        1 + SUGGESTION_SAFETY_MARGIN_PERCENT
                    )

                    # Round up to nearest multiple of 100 for a cleaner, more user-friendly suggestion
                    # E.g., 437 tokens with margin becomes 500 rather than 437
                    suggested_max_t = _round_up_to_nearest_multiple(
                        tokens_with_margin, 100
                    )

                    # STEP 2: Package the suggestion with helpful information
                    suggestions = {
                        "suggested_max_chunk_tokens": suggested_max_t,
                        "suggested_overlap_sentences": None,  # Overlap is not relevant in this case
                        "message": (
                            f"The current 'max_chunk_tokens' ({max_chunk_tokens}) is too low for at least one sentence. "
                            f"A more robust 'max_chunk_tokens' to try would be around {suggested_max_t}."
                        ),
                    }
                    raise StrictChunkingError(
                        f"Strict mode: {reason}",
                        details={
                            "type": "single_sentence_too_large",
                            "sentence_id": sentence["id"],
                            "sentence_tokens": sentence["token_count"],
                            "current_max_chunk_tokens": max_chunk_tokens,
                        },
                        suggestions=suggestions,
                    )
                else:
                    # NON-STRICT MODE: Process the oversized sentence anyway, with warning details
                    logging.warning(
                        f"Creating oversized chunk for single sentence: {reason}"
                    )

                    # Prepare overflow details to include in the chunk
                    overflow_details_for_single = reason

                    # If this chunk also had a previous overlap issue, note both problems
                    if current_chunk_overflow_reason_from_overlap:
                        overflow_details_for_single = f"{reason} (Also started with oversized overlap: {current_chunk_overflow_reason_from_overlap})"

                    # Add the oversized sentence as its own chunk with overflow details
                    chunks.append(
                        {
                            "text": sentence["text"],
                            "token_count": sentence["token_count"],
                            "id": chunk_id_counter,
                            "original_sentence_ids": [sentence["id"]],
                            "overflow_details": overflow_details_for_single,  # Include details about why this chunk exceeds limits
                        }
                    )

                    # Prepare for the next chunk
                    chunk_id_counter += 1
                    current_chunk_sentences = []  # Clear the current chunk
                    current_chunk_tokens = 0
                    current_chunk_overflow_reason_from_overlap = (
                        None  # Reset overflow flag since this chunk is handled
                    )

        # Add any remaining sentences as the final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(s["text"] for s in current_chunk_sentences)
            final_chunk_dict = {
                "text": chunk_text,
                "token_count": current_chunk_tokens,
                "id": chunk_id_counter,
                "original_sentence_ids": [s["id"] for s in current_chunk_sentences],
            }
            if current_chunk_overflow_reason_from_overlap and not strict_mode:
                final_chunk_dict["overflow_details"] = (
                    current_chunk_overflow_reason_from_overlap
                )
            chunks.append(final_chunk_dict)

        return chunks
    except StrictChunkingError:
        # Re-raise StrictChunkingError to be handled by the endpoint
        raise
    except (ValueError, TypeError) as e:
        # Re-raise these as they carry specific error information
        logging.error(f"Error in _chunk_sentences_by_token_limit: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in _chunk_sentences_by_token_limit: {str(e)}")
        raise RuntimeError(f"Error processing chunks: {str(e)}")


@app.get("/", tags=["Status"])
async def health_check():
    """Check if the API is running properly and return status information.

    Returns:
        dict: API status info with health status, version, GPU availability, and models
    """
    return {
        "status": "healthy",
        "version": app.version,
        "gpu_available": torch.cuda.is_available(),
        "default_model": DEFAULT_SAT_MODEL_NAME,
    }


@app.post("/split-sentences/", response_model=ChunkingResult, tags=["Chunking"])
async def split_sentences_endpoint(
    file: UploadFile = File(
        ..., description="Text file (.txt or .md) to split into sentences"
    ),
    input_data: SplitSentencesInput = Depends(),
):
    """Split text file into sentences using WTPSplit's advanced segmentation.

    Processes text file and returns individual sentences as chunks with metadata.
    Preserves paragraph breaks while treating single line breaks as spaces.
    Supports .txt and .md file formats.

    Returns:
        ChunkingResult: List of sentences as chunks with metadata
    """
    start_time = time.time()

    # Validate model_name
    if input_data.model_name not in VALID_SAT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name: '{input_data.model_name}'. Valid options: {sorted(VALID_SAT_MODELS)}",
        )

    # Validate file type
    if not file.filename.lower().endswith((".txt", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .txt and .md files are supported.",
        )

    try:
        # Read and decode the file content
        content = await file.read()
        text = content.decode("utf-8")

        # Preprocess the text to ignore single line breaks
        text = preprocess_text(text)

        # Split the text into sentences
        sentences = await run_in_threadpool(
            lambda: split_sentences_NLP(
                text,
                model_name=input_data.model_name,
                split_threshold=input_data.split_threshold,
            )
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create chunks from sentences and calculate token statistics
        chunks = []
        total_tokens = 0
        max_tokens = 0
        min_tokens = float("inf") if sentences else 0

        for i, sentence in enumerate(sentences):
            token_count = await run_in_threadpool(count_tokens, sentence)
            chunks.append(Chunk(text=sentence, token_count=token_count, id=i + 1))
            total_tokens += token_count
            if token_count > max_tokens:
                max_tokens = token_count
            if token_count < min_tokens:
                min_tokens = token_count

        if not sentences:
            min_tokens = 0  # Ensure min_tokens is 0 if there are no sentences

        # Prepare metadata
        metadata = ChunkingMetadata(
            n_sentences=len(chunks),
            avg_tokens_per_sentence=int(total_tokens / len(chunks)) if chunks else 0,
            max_tokens_in_sentence=max_tokens,
            min_tokens_in_sentence=min_tokens,
            sat_model_name=input_data.model_name,
            split_threshold=input_data.split_threshold,
            source=file.filename,
            processing_time=round(processing_time, 4),
        )

        return ChunkingResult(chunks=chunks, metadata=metadata)

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode file. Please ensure it's a valid UTF-8 encoded text file.",
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}",
        )


@app.post("/file-chunker/", response_model=FileChunkingResult, tags=["Chunking"])
async def file_chunker_endpoint(
    file: UploadFile = File(..., description="Text file (.txt or .md) to chunk"),
    input_data: FileChunkerInput = Depends(),
):
    """Split text into token-limited chunks with optional sentence overlap.

    First splits text into sentences, then groups them into chunks based on token limits.
    Provides comprehensive error feedback in strict mode when limits cannot be met.

    Args:
        file: Text file (.txt or .md) to process
        model_name: SaT model for sentence splitting
        split_threshold: Confidence threshold for sentence boundaries (0.0-1.0)
        max_chunk_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        strict_mode: If True, returns error when chunks exceed limits

    Returns:
        ChunkingResult: List of chunks with comprehensive metadata

    Raises:
        HTTPException (400): For invalid file types or when strict mode limits are exceeded
        HTTPException (500): For unexpected server errors
    """
    start_time = time.time()

    # Validate model_name
    if input_data.model_name not in VALID_SAT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name: '{input_data.model_name}'. Valid options: {sorted(VALID_SAT_MODELS)}",
        )

    logger.info(
        f"Processing file {file.filename} with model={input_data.model_name}, "
        f"threshold={input_data.split_threshold}, max_tokens={input_data.max_chunk_tokens}, "
        f"overlap={input_data.overlap_sentences}, strict_mode={input_data.strict_mode}"
    )

    # Validate file type
    if not file.filename.lower().endswith((".txt", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .txt and .md files are supported.",
        )

    try:
        # Read and decode file
        content = await file.read()
        text = content.decode("utf-8")
        text = preprocess_text(text)

        # Split into sentences using SaT
        sentences = await run_in_threadpool(
            lambda: split_sentences_NLP(
                text,
                model_name=input_data.model_name,
                split_threshold=input_data.split_threshold,
            )
        )

        if not sentences:
            # Handle empty input
            metadata = FileChunkingMetadata(
                split_threshold=input_data.split_threshold,
                configured_max_chunk_tokens=input_data.max_chunk_tokens,
                configured_overlap_sentences=input_data.overlap_sentences,
                n_input_sentences=0,
                avg_tokens_per_input_sentence=0,
                max_tokens_in_input_sentence=0,
                min_tokens_in_input_sentence=0,
                n_chunks=0,
                avg_tokens_per_chunk=0,
                max_tokens_in_chunk=0,
                min_tokens_in_chunk=0,
                sat_model_name=input_data.model_name,
                source=file.filename,
                processing_time=round(time.time() - start_time, 4),
            )
            return FileChunkingResult(chunks=[], metadata=metadata)

        # Get token counts for each sentence
        sentences_data = []
        total_input_tokens = 0
        max_input_tokens = 0
        min_input_tokens = float("inf")

        for i, sentence in enumerate(sentences):
            token_count = await run_in_threadpool(count_tokens, sentence)
            sentences_data.append(
                {"text": sentence, "token_count": token_count, "id": i + 1}
            )
            total_input_tokens += token_count
            max_input_tokens = max(max_input_tokens, token_count)
            min_input_tokens = min(min_input_tokens, token_count)

        min_input_tokens = min_input_tokens if sentences_data else 0
        avg_input_tokens = (
            int(total_input_tokens / len(sentences_data)) if sentences_data else 0
        )

        # Group sentences into chunks
        try:
            chunks_data = await _chunk_sentences_by_token_limit(
                sentences_data,
                input_data.max_chunk_tokens,
                input_data.overlap_sentences,
                input_data.strict_mode,
            )
        except StrictChunkingError as e:
            logger.warning(f"Strict mode chunking failed for {file.filename}: {str(e)}")

            # Create a minimal, structured error response
            minimal_response = {
                "chunk_process": "failed",
                "single_sentence_too_large": e.details.get("type")
                == "single_sentence_too_large",
                "overlap_too_large": e.details.get("type") == "overlap_too_large",
                "suggested_token_limit": e.suggestions.get("suggested_max_chunk_tokens")
                if e.suggestions
                else None,
                "suggested_overlap_value": e.suggestions.get(
                    "suggested_overlap_sentences"
                ),
            }

            # Log the full details for server-side debugging
            logger.info(
                f"Detailed chunking failure info: {e.details}, suggestions: {e.suggestions}"
            )

            raise HTTPException(
                status_code=400,  # Bad Request
                detail=minimal_response,
            )

        # Convert chunk data to Chunk objects and calculate output statistics
        chunks = []
        total_output_tokens = 0
        max_output_tokens = 0
        min_output_tokens = float("inf")

        for chunk_data in chunks_data:
            chunks.append(
                Chunk(
                    text=chunk_data["text"],
                    token_count=chunk_data["token_count"],
                    id=chunk_data["id"],
                    overflow_details=chunk_data.get(
                        "overflow_details"
                    ),  # Include overflow_details if present
                )
            )
            total_output_tokens += chunk_data["token_count"]
            max_output_tokens = max(max_output_tokens, chunk_data["token_count"])
            min_output_tokens = min(min_output_tokens, chunk_data["token_count"])

        min_output_tokens = min_output_tokens if chunks else 0
        avg_output_tokens = int(total_output_tokens / len(chunks)) if chunks else 0

        # Create metadata
        metadata = FileChunkingMetadata(
            split_threshold=input_data.split_threshold,
            configured_max_chunk_tokens=input_data.max_chunk_tokens,
            configured_overlap_sentences=input_data.overlap_sentences,
            n_input_sentences=len(sentences_data),
            avg_tokens_per_input_sentence=avg_input_tokens,
            max_tokens_in_input_sentence=max_input_tokens,
            min_tokens_in_input_sentence=min_input_tokens,
            n_chunks=len(chunks),
            avg_tokens_per_chunk=avg_output_tokens,
            max_tokens_in_chunk=max_output_tokens,
            min_tokens_in_chunk=min_output_tokens,
            sat_model_name=input_data.model_name,
            source=file.filename,
            processing_time=round(time.time() - start_time, 4),
        )

        return FileChunkingResult(chunks=chunks, metadata=metadata)

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode file. Please ensure it's a valid UTF-8 encoded text file.",
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}",
        )
