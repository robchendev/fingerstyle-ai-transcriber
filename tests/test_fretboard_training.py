import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.fretboard_training import (
    DetectorTrainingConfig, training_request, validate_dataset,
)


class FretboardTrainingTests(unittest.TestCase):
    def dataset(self, root):
        keypoints = ["a", "b", "c", "d", "e", "f"]
        records = []
        annotations = {"keypoints": keypoints}
        for index, split in enumerate(("train", "validation")):
            identifier = f"frame-{index}"
            image = root / "images" / split / f"{identifier}.jpg"
            label = root / "labels" / split / f"{identifier}.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            label.write_text("0 .5 .5 .8 .4 " + " ".join((".2 .3 2",) * 6) + "\n")
            records.append({
                "id": identifier, "split": split,
                "image": str(image.relative_to(root)).replace("\\", "/"),
                "sourceVideoSha256": str(index) * 64,
            })
        (root / "manifest.json").write_text(json.dumps({"keypoints": keypoints, "records": records}))
        (root / "annotations.json").write_text(json.dumps(annotations))
        (root / "data.yaml").write_text("kpt_shape: [6, 3]\n")
        return root

    def test_validates_split_dataset_and_builds_4k_request(self):
        with TemporaryDirectory() as directory:
            root = self.dataset(Path(directory))
            value = validate_dataset(root)
            self.assertEqual(value["counts"], {"train": 1, "validation": 1, "test": 0})
            config = DetectorTrainingConfig()
            request = training_request(root, root / "run", config)
            self.assertEqual(request["config"]["image_size"], 3840)
            self.assertEqual(request["config"]["batch_size"], 1)
            self.assertEqual(len(request["datasetSha256"]), 64)

    def test_rejects_video_split_leakage_and_invalid_labels(self):
        with TemporaryDirectory() as directory:
            root = self.dataset(Path(directory))
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["records"][1]["sourceVideoSha256"] = manifest["records"][0]["sourceVideoSha256"]
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "crosses"):
                validate_dataset(root)

    def test_config_rejects_invalid_training_limits(self):
        for options in (
            {"image_size": 319}, {"batch_size": 0}, {"epochs": 0},
            {"workers": -1}, {"device": ""},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DetectorTrainingConfig(**options)


if __name__ == "__main__":
    unittest.main()
