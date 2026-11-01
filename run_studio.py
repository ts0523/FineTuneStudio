# -*- coding: utf-8 -*-
"""FineTune Studio launcher (source run and PyInstaller build).

Source:  .venv\\Scripts\\pythonw.exe run_studio.py
Build:   FineTuneStudio.exe
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from studio.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
