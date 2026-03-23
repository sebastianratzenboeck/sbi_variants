from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
REPO_PARENT = REPO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
