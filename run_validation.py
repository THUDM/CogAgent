#!/usr/bin/env python3
"""
Simple validation script to check testing infrastructure setup.
This can be run without Poetry installation to verify the structure.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print result."""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print result."""
    exists = Path(dirpath).exists() and Path(dirpath).is_dir()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {dirpath}")
    return exists

def check_file_contains(filepath, search_text, description):
    """Check if a file contains specific text."""
    try:
        content = Path(filepath).read_text()
        contains = search_text in content
        status = "✓" if contains else "✗"
        print(f"{status} {description}")
        return contains
    except:
        print(f"✗ {description} (file not readable)")
        return False

def main():
    """Run validation checks."""
    print("Testing Infrastructure Validation")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check project structure
    print("\n1. Project Structure:")
    all_checks_passed &= check_file_exists("pyproject.toml", "Poetry configuration file")
    all_checks_passed &= check_directory_exists("tests", "Tests directory")
    all_checks_passed &= check_directory_exists("tests/unit", "Unit tests directory")
    all_checks_passed &= check_directory_exists("tests/integration", "Integration tests directory")
    all_checks_passed &= check_file_exists("tests/__init__.py", "Tests package init")
    all_checks_passed &= check_file_exists("tests/unit/__init__.py", "Unit tests init")
    all_checks_passed &= check_file_exists("tests/integration/__init__.py", "Integration tests init")
    all_checks_passed &= check_file_exists("tests/conftest.py", "Pytest configuration")
    all_checks_passed &= check_file_exists("tests/test_setup_validation.py", "Validation tests")
    
    # Check pyproject.toml contents
    print("\n2. Poetry Configuration:")
    all_checks_passed &= check_file_contains("pyproject.toml", "[tool.poetry]", "Poetry section")
    all_checks_passed &= check_file_contains("pyproject.toml", "[tool.poetry.dependencies]", "Dependencies section")
    all_checks_passed &= check_file_contains("pyproject.toml", "[tool.poetry.group.dev.dependencies]", "Dev dependencies")
    all_checks_passed &= check_file_contains("pyproject.toml", "pytest", "Pytest dependency")
    all_checks_passed &= check_file_contains("pyproject.toml", "pytest-cov", "Coverage dependency")
    all_checks_passed &= check_file_contains("pyproject.toml", "pytest-mock", "Mock dependency")
    
    # Check pytest configuration
    print("\n3. Pytest Configuration:")
    all_checks_passed &= check_file_contains("pyproject.toml", "[tool.pytest.ini_options]", "Pytest config section")
    all_checks_passed &= check_file_contains("pyproject.toml", "[tool.coverage", "Coverage config section")
    all_checks_passed &= check_file_contains("pyproject.toml", 'test = "pytest"', "Test script command")
    all_checks_passed &= check_file_contains("pyproject.toml", 'tests = "pytest"', "Tests script command")
    
    # Check test markers
    print("\n4. Test Markers:")
    all_checks_passed &= check_file_contains("pyproject.toml", '"unit: Unit tests"', "Unit test marker")
    all_checks_passed &= check_file_contains("pyproject.toml", '"integration: Integration tests"', "Integration test marker")
    all_checks_passed &= check_file_contains("pyproject.toml", '"slow: Slow running tests"', "Slow test marker")
    
    # Check fixtures
    print("\n5. Test Fixtures:")
    all_checks_passed &= check_file_contains("tests/conftest.py", "def temp_dir", "Temp directory fixture")
    all_checks_passed &= check_file_contains("tests/conftest.py", "def mock_model_config", "Model config fixture")
    all_checks_passed &= check_file_contains("tests/conftest.py", "def mock_tokenizer", "Tokenizer fixture")
    
    # Check .gitignore updates
    print("\n6. Git Configuration:")
    all_checks_passed &= check_file_contains(".gitignore", ".claude/*", "Claude settings ignored")
    all_checks_passed &= check_file_contains(".gitignore", "# NOTE: We DO NOT ignore poetry.lock", "Poetry lock note")
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✓ All validation checks passed!")
        print("\nNext steps:")
        print("1. Run: poetry install --with dev")
        print("2. Run: poetry run test")
        print("3. Run: poetry run tests")
        return 0
    else:
        print("✗ Some validation checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())