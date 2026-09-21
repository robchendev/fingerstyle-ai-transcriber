"""Video preparation regressions using the isolated video dependencies."""

from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

if any(find_spec(name) is None for name in ("av", "cv2", "mediapipe")):
    raise unittest.SkipTest("Run video tests with the isolated vision environment.")

VIDEO_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "video-evidence"
if str(VIDEO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(VIDEO_SCRIPTS))
