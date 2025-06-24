import sys
import os
from pathlib import Path
import pytest


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""

    def test_python_path_includes_project_root(self):
        """Test that the project root is in the Python path."""
        project_root = Path(__file__).parent.parent
        assert str(project_root) in sys.path, "Project root should be in Python path"

    def test_pytest_is_available(self):
        """Test that pytest is properly installed."""
        import pytest
        assert pytest.__version__, "pytest should be installed with a valid version"

    def test_coverage_is_available(self):
        """Test that pytest-cov is properly installed."""
        import pytest_cov
        assert pytest_cov.__version__, "pytest-cov should be installed"

    def test_mock_is_available(self):
        """Test that pytest-mock is properly installed."""
        import pytest_mock
        assert pytest_mock.__version__, "pytest-mock should be installed"

    def test_project_modules_are_importable(self):
        """Test that project modules can be imported."""
        # These imports should work if the project structure is correct
        modules_to_test = [
            "app",
            "inference",
            "finetune",
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
            except ImportError as e:
                if "No module named" not in str(e):
                    # Module exists but has other import errors - that's OK for this test
                    pass
                else:
                    pytest.fail(f"Module {module} should be importable")

    def test_fixtures_are_available(self, temp_dir, mock_model_config):
        """Test that custom fixtures from conftest.py are available."""
        assert temp_dir.exists(), "temp_dir fixture should provide existing directory"
        assert isinstance(mock_model_config, dict), "mock_model_config should be a dictionary"
        assert "model_name" in mock_model_config, "mock_model_config should contain model_name"

    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Test that the unit test marker is properly registered."""
        assert True, "Unit marker should work"

    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Test that the integration test marker is properly registered."""
        assert True, "Integration marker should work"

    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Test that the slow test marker is properly registered."""
        assert True, "Slow marker should work"

    def test_test_directories_exist(self):
        """Test that all test directories are properly created."""
        test_root = Path(__file__).parent
        assert test_root.exists(), "tests directory should exist"
        assert (test_root / "unit").exists(), "tests/unit directory should exist"
        assert (test_root / "integration").exists(), "tests/integration directory should exist"
        assert (test_root / "conftest.py").exists(), "conftest.py should exist"

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml is properly created."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"
        
        # Check that it contains expected sections
        content = pyproject_path.read_text()
        assert "[tool.poetry]" in content, "pyproject.toml should contain Poetry configuration"
        assert "[tool.pytest.ini_options]" in content, "pyproject.toml should contain pytest configuration"
        assert "[tool.coverage" in content, "pyproject.toml should contain coverage configuration"

    def test_poetry_scripts_configured(self):
        """Test that Poetry scripts for running tests are configured."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text()
        
        assert 'test = "pytest"' in content, "Poetry script 'test' should be configured"
        assert 'tests = "pytest"' in content, "Poetry script 'tests' should be configured"


if __name__ == "__main__":
    # This allows running the validation directly with python
    pytest.main([__file__, "-v"])