"""
pytest configuration file.

This file automatically adds the project root directory to the Python path
so that the gaussian_elimination package can be imported during testing.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root)) 