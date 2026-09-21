import subprocess
import sys
import unittest

from scripts import video_features
from scripts.dataset_io import ROOT


class VideoContractTests(unittest.TestCase):
    def test_contract_partitions_all_features_and_imports_without_torch(self):
        self.assertEqual(video_features.SCHEMA_VERSION, 4)
        self.assertEqual(video_features.INPUT_REPRESENTATION, "guitar-hand-coarse-194-v1")
        self.assertEqual(video_features.STRUCTURED_DIM, 194)
        self.assertEqual(video_features.VIEW_ORDER, ["fretting", "plucking", "unassigned_0", "unassigned_1"])
        covered = []
        for group in (video_features.OBSERVATION_SLICES, video_features.VELOCITY_SLICES):
            for start, stop in group:
                covered.extend(range(start, stop))
        self.assertEqual(sorted(covered), list(range(194)))
        self.assertEqual(video_features.VELOCITY_SLICES[-1], (188, 190))
        self.assertEqual(video_features.OBSERVATION_SLICES[-2:], ((186, 188), (190, 194)))
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from scripts import video_features; assert 'torch' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
