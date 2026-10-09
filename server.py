"""
Hindi AI Calling System — Entry Point.
Directs to the modular, production-structured backend package in backend/.
Run:
    python server.py
    or
    python backend/run.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app import app
from backend.run import main

if __name__ == "__main__":
    main()