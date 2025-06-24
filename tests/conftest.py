import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
from unittest.mock import Mock, MagicMock

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Provide a temporary file within a temporary directory."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("Test content")
    yield temp_file_path


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Provide a mock configuration for model testing."""
    return {
        "model_name": "cogagent-test",
        "model_path": "/path/to/model",
        "device": "cpu",
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "num_beams": 1,
    }


@pytest.fixture
def mock_transformers_model() -> Mock:
    """Provide a mock transformers model."""
    model = MagicMock()
    model.generate.return_value = MagicMock()
    model.config = MagicMock(
        max_position_embeddings=2048,
        hidden_size=768,
        num_attention_heads=12,
    )
    return model


@pytest.fixture
def mock_tokenizer() -> Mock:
    """Provide a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "Mock decoded text"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    return tokenizer


@pytest.fixture
def mock_image_data() -> bytes:
    """Provide mock image data for testing."""
    # Simple 1x1 PNG image
    return (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00'
        b'\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\r'
        b'IDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )


@pytest.fixture
def mock_openai_client() -> Mock:
    """Provide a mock OpenAI client."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Mock response"))]
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_gradio_interface() -> Mock:
    """Provide a mock Gradio interface."""
    interface = MagicMock()
    interface.launch.return_value = None
    return interface


@pytest.fixture
def env_setup(monkeypatch) -> None:
    """Set up test environment variables."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/tmp/test_cache")
    monkeypatch.setenv("HF_HOME", "/tmp/test_hf_home")


@pytest.fixture
def mock_torch_cuda(monkeypatch) -> None:
    """Mock torch CUDA availability for CPU testing."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture(autouse=True)
def cleanup_imports():
    """Clean up imports after each test to avoid side effects."""
    modules_to_remove = [
        mod for mod in sys.modules 
        if mod.startswith(('app.', 'inference.', 'finetune.'))
    ]
    yield
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)


@pytest.fixture
def capture_logs(caplog):
    """Capture and return logs for testing."""
    with caplog.at_level("DEBUG"):
        yield caplog


@pytest.fixture
def mock_subprocess_run(monkeypatch) -> Mock:
    """Mock subprocess.run for testing command execution."""
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Success"
    mock_run.return_value.stderr = ""
    monkeypatch.setattr("subprocess.run", mock_run)
    return mock_run


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


# Hooks for test execution
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)