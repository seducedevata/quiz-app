"""
Pytest configuration for enterprise-grade clean test suite
"""
import pytest
import tempfile
import shutil
import warnings
import os
import sys
from pathlib import Path
from unittest.mock import Mock
import torch

# ENTERPRISE WARNING SUPPRESSION - Maximum suppression for clean test output
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', PendingDeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', ImportWarning)
warnings.simplefilter('ignore', ResourceWarning)
warnings.simplefilter('ignore', UserWarning)

# Specific PyQt warning suppression
warnings.filterwarnings('ignore', message=r'.*sipPyTypeDict.*')
warnings.filterwarnings('ignore', message=r'.*SwigPy.*')
warnings.filterwarnings('ignore', message=r'.*swigvarlink.*')
warnings.filterwarnings('ignore', message=r'builtin type.*has no __module__ attribute')

def pytest_configure(config):
    """Configure pytest with maximum warning suppression."""
    warnings.filterwarnings('ignore')
    print("Enterprise warning suppression activated")

def pytest_sessionstart(session):
    """Suppress warnings at session start."""
    warnings.filterwarnings('ignore')

def pytest_runtest_setup(item):
    """Suppress warnings for each test."""
    warnings.filterwarnings('ignore')

# Test fixtures for common test dependencies

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def mock_config():
    """Mock configuration for tests"""
    config = Mock()
    config.get_value = Mock(return_value="test_value")
    config.set_value = Mock()
    config.save = Mock()
    return config

@pytest.fixture
def sample_data_dir(temp_dir):
    """Create sample data directory structure"""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    # Create sample files
    (data_dir / "sample.txt").write_text("Sample text content")
    (data_dir / "sample.json").write_text('{"test": "data"}')
    
    return data_dir

@pytest.fixture
def torch_device():
    """Get appropriate torch device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
