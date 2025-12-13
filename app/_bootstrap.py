# app/_bootstrap.py
# Ensure project root is on sys.path so "import core.*" works from Streamlit pages.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
