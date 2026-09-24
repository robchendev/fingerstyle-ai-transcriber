import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.fretboard_annotation import (
    HTML,
    KEYPOINTS,
    export_yolo,
    normalize_annotation,
    normalize_preferences,
    validate_geometry,
)


def annotation():
    coordinates = (
        (.30, .20), (.30, .25),
        (.50, .18), (.50, .27),
        (.85, .12), (.85, .33),
    )
    return {
        "points": [
            {"x": x, "y": y, "visibility": 2}
            for x, y in coordinates
        ],
        "complete": True,
        "note": "",
    }


class FretboardAnnotationTests(unittest.TestCase):
    def test_ui_uses_human_landmark_names_and_prominent_save_state(self):
        self.assertIn("Bridge — High E / String 1", HTML)
        self.assertIn("12th fret — Low E / String 6", HTML)
        self.assertIn('id="saveState"', HTML)
        self.assertIn('id="progress"', HTML)
        self.assertIn('e.key=="Enter"', HTML)
        self.assertIn('e.key=="ArrowLeft"', HTML)
        self.assertIn('e.key=="ArrowRight"', HTML)
        self.assertIn('"complete";if(value.note.trim()', HTML)
        self.assertIn('id="opacity"', HTML)
        self.assertIn('fetch("/api/preferences"', HTML)
        self.assertIn("ctx.globalAlpha=overlayOpacity", HTML)
        self.assertIn('id="dotRadius"', HTML)
        self.assertIn("ctx.arc(X,Y,dotRadius,0", HTML)
        self.assertIn('if(i==selected){ctx.strokeStyle="#000"', HTML)
        self.assertNotIn("dotRadius+(i==selected", HTML)
        self.assertIn("function placeAt(e)", HTML)
        self.assertIn("else if(placing){placeAt(e)}", HTML)
        self.assertNotIn("selectPoint(Math.min(5,selected+1))", HTML)
        self.assertIn("ctx.moveTo(left,cursorY)", HTML)
        self.assertIn("ctx.moveTo(cursorX,top)", HTML)
        self.assertIn('ctx.lineWidth=1/devicePixelRatio;ctx.strokeStyle="#FFF"', HTML)
        self.assertNotIn('ctx.lineWidth=3;ctx.strokeStyle="rgba(0,0,0,.7)"', HTML)
        self.assertIn("background:#fff;color:#111", HTML)
        self.assertNotIn("background:#111;color:#eee", HTML)
        self.assertIn(".pointButton.available{background:#e6f4ea}", HTML)
        self.assertIn(".pointButton.occluded{background:#fff4ce}", HTML)
        self.assertIn(".pointButton.unavailable{background:#fde7e9}", HTML)
        self.assertIn("<strong>Available:</strong>", HTML)
        self.assertIn("<strong>Occluded:</strong>", HTML)
        self.assertIn("<strong>Unavailable:</strong>", HTML)
        self.assertIn("It has no effect on training.", HTML)
        self.assertIn("Available (Q)", HTML)
        self.assertIn("Occluded (W)", HTML)
        self.assertIn("Unavailable (E)", HTML)
        self.assertIn("function setSelectedStatus(value)", HTML)
        self.assertIn("point.visibility=value", HTML)
        self.assertIn('$("clear").classList.toggle("active",mode==0)', HTML)
        self.assertIn("setMode(item().points[selected].visibility)", HTML)
        self.assertIn('if(mode==0){setStatus("Choose Available (Q) or Occluded (W)', HTML)
        self.assertIn('e.key.toLowerCase()=="q"', HTML)
        self.assertIn('e.key.toLowerCase()=="w"', HTML)
        self.assertIn('e.key.toLowerCase()=="e"', HTML)
        self.assertIn('id="captureArrows"', HTML)
        self.assertIn('&&captureArrows){e.preventDefault();e.stopPropagation()', HTML)
        self.assertIn("document.activeElement.blur()", HTML)
        self.assertNotIn("localStorage", HTML)
        self.assertNotIn("`${i+1}: ${n}`", HTML)

    def test_preferences_are_validated_for_local_file_storage(self):
        expected = {
            "schemaVersion": 1,
            "kind": "fretboard-annotation-preferences",
            "overlayOpacity": .45,
            "captureArrowKeys": True,
            "dotRadius": 7.,
        }
        self.assertEqual(normalize_preferences(expected), expected)
        for changed in (
            {**expected, "overlayOpacity": 0},
            {**expected, "captureArrowKeys": "yes"},
            {**expected, "dotRadius": 21},
            {**expected, "extra": True},
        ):
            with self.assertRaises(ValueError):
                normalize_preferences(changed)

    def test_six_point_contract_normalizes_complete_geometry(self):
        value = normalize_annotation(annotation())
        self.assertEqual(len(value["points"]), 6)
        self.assertTrue(value["complete"])
        validate_geometry(value["points"])

    def test_complete_geometry_allows_unavailable_out_of_frame_anchors(self):
        value = annotation()
        value["points"][4:] = [
            {"x": None, "y": None, "visibility": 0},
            {"x": None, "y": None, "visibility": 0},
        ]
        normalized = normalize_annotation(value)
        self.assertTrue(normalized["complete"])
        self.assertEqual([point["visibility"] for point in normalized["points"]], [2, 2, 2, 2, 0, 0])

    def test_complete_geometry_rejects_flipped_and_degenerate_points(self):
        value = annotation()
        value["points"][2], value["points"][3] = value["points"][3], value["points"][2]
        with self.assertRaisesRegex(ValueError, "ordering"):
            normalize_annotation(value)
        value = annotation()
        value["points"][1] = dict(value["points"][0])
        with self.assertRaisesRegex(ValueError, "too close"):
            normalize_annotation(value)

    def test_yolo_export_uses_six_keypoints_and_preserves_incomplete_items(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "images" / "train" / "frame.jpg").write_bytes(b"jpeg")
            (root / "manifest.json").write_text(json.dumps({
                "schemaVersion": 1,
                "kind": "fretboard-keypoint-annotation-dataset",
                "keypoints": list(KEYPOINTS),
                "records": [{"id": "frame", "split": "train", "image": "images/train/frame.jpg"}],
            }))
            (root / "annotations.json").write_text(json.dumps({
                "schemaVersion": 1,
                "kind": "fretboard-keypoint-annotations",
                "keypoints": list(KEYPOINTS),
                "items": {"frame": annotation()},
            }))
            self.assertEqual(export_yolo(root), {"train": 1, "validation": 0, "test": 0})
            fields = (root / "labels" / "train" / "frame.txt").read_text().split()
            self.assertEqual(len(fields), 5 + 6 * 3)
            self.assertIn("kpt_shape: [6, 3]", (root / "data.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
