"""
run.py — One-command launcher for the Hallucination Detection API.

Usage:
    python run.py

The server starts at http://localhost:8000
Interactive API docs: http://localhost:8000/docs
"""

import os
import uvicorn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if __name__ == "__main__":
    uvicorn.run(
        "Backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
