# Full corrected content for test_api.py
import pytest
import sys
import os
from fastapi.testclient import TestClient

# Ensure the app module is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentences_chunker import app  # Import the FastAPI app instance

# Directory for test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create test data directory and ensure the test file exists before tests run."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    raptor_path = os.path.join(TEST_DATA_DIR, "raptor.md")

    # Verify that the test file exists
    if not os.path.exists(raptor_path):
        raise FileNotFoundError(f"Required test file not found: {raptor_path}")

    yield  # Allow tests to run


@pytest.fixture
def client():
    """Provides a TestClient instance for making requests to the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test the root endpoint ('/') returns 200 with the correct status and version."""
    response = client.get("/")
    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    )
    data = response.json()
    assert "status" in data, "Response JSON missing 'status' key"
    assert data["status"] == "healthy", (
        f"Expected status 'healthy', got '{data.get('status')}'"
    )
    assert "version" in data, "Response JSON missing 'version' key"


def test_standard_chunking(client):
    """Test the standard chunking endpoint ('/file-chunker') and validate response structure."""
    raptor_path = os.path.join(TEST_DATA_DIR, "raptor.md")
    with open(raptor_path, "rb") as f:
        response = client.post(
            "/file-chunker",  # Correct endpoint path
            files={"file": ("raptor.md", f, "text/plain")},
            # Use the correct parameter name 'max_chunk_tokens'
            params={"max_chunk_tokens": 600, "overlap": 0},
        )
    assert response.status_code == 200, f"API returned error: {response.text}"
    data = response.json()

    # Validate top-level structure
    assert "chunks" in data, "Response missing 'chunks' key"
    assert "metadata" in data, "Response missing 'metadata' key"

    # Validate chunks
    chunks = data["chunks"]
    assert isinstance(chunks, list), "Chunks should be a list"
    assert len(chunks) > 0, "Should have at least one chunk"
    for i, chunk in enumerate(chunks):
        assert "text" in chunk, f"Chunk {i} missing 'text' field"
        assert "token_count" in chunk, f"Chunk {i} missing 'token_count' field"
        assert "id" in chunk, f"Chunk {i} missing 'id' field"
        assert isinstance(chunk["text"], str) and len(chunk["text"]) > 0, (
            f"Chunk {i} text is invalid"
        )
        assert isinstance(chunk["token_count"], int) and chunk["token_count"] > 0, (
            f"Chunk {i} token_count is invalid"
        )
        # Check if token count respects the limit
        assert chunk["token_count"] <= 600, (
            f"Chunk {i} token count {chunk['token_count']} exceeds max_chunk_tokens 600"
        )
        assert isinstance(chunk["id"], int) and chunk["id"] == i + 1, (
            f"Chunk {i} ID is invalid or not sequential"
        )

    # Validate metadata
    metadata = data["metadata"]
    expected_meta_keys = [
        "file",
        "n_chunks",
        "avg_tokens",
        "max_tokens",
        "min_tokens",
        "max_chunk_tokens",
        "overlap",
        "processing_time",
    ]
    for key in expected_meta_keys:
        assert key in metadata, f"Metadata missing '{key}' field"

    assert metadata["file"] == "raptor.md", "Metadata 'file' name mismatch"
    assert metadata["n_chunks"] == len(chunks), "Metadata 'n_chunks' mismatch"
    assert metadata["max_chunk_tokens"] == 600, "Metadata 'max_chunk_tokens' mismatch"
    assert metadata["overlap"] == 0, "Metadata 'overlap' mismatch"
    assert (
        isinstance(metadata["processing_time"], float)
        and metadata["processing_time"] >= 0
    ), "Metadata 'processing_time' is invalid"


def test_adaptive_chunking(client):
    """Test the adaptive chunking endpoint ('/adaptive-file-chunking') and validate response structure."""
    raptor_path = os.path.join(TEST_DATA_DIR, "raptor.md")
    with open(raptor_path, "rb") as f:
        response = client.post(
            "/adaptive-file-chunking",  # Correct endpoint path
            files={"file": ("raptor.md", f, "text/plain")},
            params={"overlap": 1},  # Required param for adaptive chunking (1, 2, or 3)
        )
    assert response.status_code == 200, f"API returned error: {response.text}"
    data = response.json()

    # Validate top-level structure
    assert "chunks" in data, "Response missing 'chunks' key"
    assert "metadata" in data, "Response missing 'metadata' key"

    # Validate chunks
    chunks = data["chunks"]
    assert isinstance(chunks, list), "Chunks should be a list"
    assert len(chunks) > 0, "Should have at least one chunk"
    for i, chunk in enumerate(chunks):
        assert "text" in chunk, f"Chunk {i} missing 'text' field"
        assert "token_count" in chunk, f"Chunk {i} missing 'token_count' field"
        assert "id" in chunk, f"Chunk {i} missing 'id' field"
        assert isinstance(chunk["text"], str) and len(chunk["text"]) > 0, (
            f"Chunk {i} text is invalid"
        )
        assert isinstance(chunk["token_count"], int) and chunk["token_count"] > 0, (
            f"Chunk {i} token_count is invalid"
        )
        assert isinstance(chunk["id"], int) and chunk["id"] == i + 1, (
            f"Chunk {i} ID is invalid or not sequential"
        )
        # We don't check token_count against a fixed limit here, as it's dynamic

    # Validate metadata
    metadata = data["metadata"]
    expected_meta_keys = [
        "file",
        "n_chunks",
        "avg_tokens",
        "max_tokens",
        "min_tokens",
        "max_chunk_tokens",
        "overlap",
        "processing_time",
    ]
    for key in expected_meta_keys:
        assert key in metadata, f"Metadata missing '{key}' field"

    assert metadata["file"] == "raptor.md", "Metadata 'file' name mismatch"
    assert metadata["n_chunks"] == len(chunks), "Metadata 'n_chunks' mismatch"
    assert metadata["overlap"] == 1, (
        "Metadata 'overlap' should match the parameter value (1)"
    )
    # max_chunk_tokens is dynamically calculated, just check it's positive
    assert (
        isinstance(metadata["max_chunk_tokens"], int)
        and metadata["max_chunk_tokens"] > 0
    ), "Metadata 'max_chunk_tokens' is invalid or not positive"
    assert (
        isinstance(metadata["processing_time"], float)
        and metadata["processing_time"] >= 0
    ), "Metadata 'processing_time' is invalid"


# Note: The test for 'title_aware_chunking' is removed as the
# corresponding endpoint '/chunker/title_aware/' does not exist
# in the current 'sentences_chunker.py' implementation.
